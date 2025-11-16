---
layout: post
title: "Rust + CUDA Ray-Marching Engine"
description: "Real-time procedural geometry with SDFs, CSG trees, and space folding"
date: 2025-06-12
tags: [rust, cuda, graphics, ecs, sdl2, csg, sdf, ray marching]
toc: true
---

Building a real-time renderer that can generate complex procedural geometry entirely on the GPU—that's been my goal with this project. The result is a compact engine that combines Rust's safety with CUDA's raw performance to render scenes defined purely by mathematical functions.

This project was inspired by the pioneering work of Inigo Quilez, whose research on distance functions and ray marching techniques has been instrumental in bringing procedural rendering to real-time graphics. 

<p align="center">
  <img src="/assets/img/Xor_engine/Xor_menger.png" alt="Menger sponge" width="90%">
</p>

**Code:** [github.com/EtiNL/xor_engine](https://github.com/EtiNL/xor_engine)

## What This Engine Does

At its core, this is a **ray marching renderer** that evaluates implicit surfaces defined by signed distance functions (SDFs). Instead of storing geometry as meshes, everything exists as mathematical equations evaluated in real-time. The engine can:

- Render primitive shapes (spheres, boxes, cones, planes, lines)
- Combine them using CSG operations (union, intersection, difference)
- Repeat geometry infinitely through space folding
- Apply textures with automatic UV mapping
- Handle dynamic scene updates with minimal overhead

The entire pipeline runs on the GPU via custom CUDA kernels, orchestrated from a Rust host layer with an ECS architecture for scene management.

---

## The Core Idea: Signed Distance Functions

Traditional 3D rendering works with triangle meshes—you store vertex positions and render them. But there's another way: **implicit surfaces** defined by mathematical functions.

A **Signed Distance Function** (SDF) takes a point in 3D space and returns the shortest distance to a surface. The sign tells you which side you're on:
- **Negative** = inside the shape
- **Positive** = outside the shape  
- **Zero** = exactly on the surface

Here's a sphere SDF in its simplest form:

```rust
fn sdf_sphere(point: Vec3, center: Vec3, radius: f32) -> f32 {
    (point - center).length() - radius
}
```

That's it. No vertices, no triangles—just pure math. Query any point in space and you instantly know how far you are from the sphere's surface.

Inigo Quilez maintains an complete [library of distance functions](https://iquilezles.org/articles/distfunctions/) covering dozens of primitives—boxes, tori, cones, capsules, and far more exotic shapes.

### Why SDFs?

SDFs have some remarkable properties:

1. **Perfect geometry**: No tessellation artifacts, no LOD popping. The surface is mathematically exact at every scale.

2. **Easy boolean operations**: Want to carve a hole in something? Just use `max(shape_a, -shape_b)`. Union is `min()`, intersection is `max()`. CSG operations become trivial.

3. **Free normals**: The gradient of an SDF at any point is the surface normal. For analytic SDFs like spheres and boxes, these are exact. For others, a simple numerical gradient works.

4. **Procedural everything**: Deform space before evaluating the SDF and you get infinite repetitions, twists, bends—all essentially free.

The tradeoff? You need ray marching instead of rasterization.

---

## Ray Marching: Sphere Tracing Through Implicit Space

Traditional ray tracing shoots rays and calculates exact intersections with geometry. With SDFs, we don't have explicit surfaces—so we **march** along the ray.

Here's the algorithm (sphere tracing):

```
for each pixel:
    ray = generate_camera_ray(pixel)
    t = 0
    
    while t < max_distance:
        point = ray.origin + ray.direction * t
        distance = evaluate_sdf(point)
        
        if distance < epsilon:
            # Hit! We're close enough to the surface
            return shade(point)
        
        t += distance  # Safe step: we can move this far without hitting anything
```

The SDF tells us the minimum safe distance we can step. We can't overshoot because we know nothing is closer than that distance.

This makes sphere tracing **self-adaptive**—it takes large steps in empty space and tiny steps near surfaces, automatically handling detail.

### The Implementation

In my CUDA kernel, each pixel launches one thread that marches its ray:

```cuda
extern "C" __global__
void raymarch(/* ... scene data ... */) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    Vec3 origin = /* ... from camera ... */;
    Vec3 dir = /* ... from camera ... */;
    Vec3 p = origin;
    float total_dist = 0.0f;
    
    const float eps = 0.01f;
    const float max_dist = 200.0f;
    const int max_steps = 1000;
    
    int steps = 0;
    float min_dist;
    
    while (steps < max_steps) {
        min_dist = 1e20f;
        
        // Evaluate every SDF object and CSG tree
        for (int j = 0; j < num_objs; ++j) {
            float d = evaluate_sdf(sdf_objs[j], p);
            min_dist = fminf(min_dist, d);
        }
        
        if (min_dist < eps || total_dist > max_dist) break;
        
        p = p + dir * min_dist;
        total_dist += min_dist;
        steps++;
    }
    
    // Shade the hit point...
}
```

For each step, I evaluate all active SDF objects and take the minimum distance. If any distance falls below epsilon, we've hit a surface.

---

## Constructive Solid Geometry: Building Complex Shapes

Individual primitives are nice, but the real power comes from **combining** them. This is where CSG (Constructive Solid Geometry) enters.

### CSG Operations

With SDFs, boolean operations are just min/max functions:

- **Union** (A ∪ B): `min(sdf_a, sdf_b)`  
  Keep whichever surface is closest

- **Intersection** (A ∩ B): `max(sdf_a, sdf_b)`  
  Only keep points inside both shapes

- **Difference** (A \ B): `max(sdf_a, -sdf_b)`  
  Remove B from A by inverting B's distance field

### Binary CSG Trees

I represent CSG operations as binary trees where:
- **Leaf nodes** = primitive SDF objects (entities in the ECS)
- **Operation nodes** = Union/Intersection/Difference

For example, the classic "sphere with crossed slots" (like in my demo scene):

```
        Difference
        /        \
    Sphere      Union
                /    \
            Union   BoxZ
            /   \
        BoxX  BoxY
```

This carves three orthogonal rectangular slots through a sphere.

### GPU-Friendly CSG Evaluation

The challenge: GPUs hate dynamic recursion and branching. My solution is to **flatten the tree into arrays** that can be evaluated iteratively.

During scene updates, I traverse the tree in-order to produce:

1. **Leaf array**: The SDF objects in evaluation order `[Sphere, BoxX, BoxY, BoxZ]`
2. **Pair array**: Which leaves/intermediates combine at each step `[(BoxX,BoxY), (Union1,BoxZ), (Sphere,Union2)]`  
3. **Operation array**: What operation to use `[Union, Union, Difference]`

On the GPU, I start with distances for all leaves, then iteratively combine pairs according to the operation array:

```cuda
// Start with leaf distances in registers
float d[4] = {dist_sphere, dist_box_x, dist_box_y, dist_box_z};

// Combine according to pair array
d[0] = min(d[1], d[2]);        // Union: BoxX ∪ BoxY
d[0] = min(d[0], d[3]);        // Union: (BoxX ∪ BoxY) ∪ BoxZ  
d[0] = max(d_sphere, -d[0]);   // Difference: Sphere \ Union
```

The entire evaluation happens in registers with no recursion. For trees with N leaves, I use template metaprogramming to unroll the loop based on the next power-of-two bucket (2, 4, 8, 16, 32, 64 leaves).

For trees with more than 64 leaves that are union-only (common for scattered objects), I fall back to a simple linear min-reduction since there's no gradient tracking needed.

### Gradient Sign Tracking

Here's a subtle detail: CSG operations can flip normals. When you use **Difference**, you're inverting one operand's distance field—which also inverts its gradient.

I track a `grad_sign` multiplier through the CSG evaluation. Union and Intersection preserve it (+1), but Difference flips it for the subtracted operand (-1). When computing the final normal:

```cuda
Vec3 normal = evaluate_grad_sdf(obj, hit_point, center) * grad_sign;
```

This ensures lighting stays correct even for complex boolean combinations.

---

## Space Folding: Infinite Repetition

One of the most powerful SDF techniques is **space folding**—repeating geometry infinitely by transforming space before evaluating the SDF.

### The Concept

Instead of evaluating `sdf(p)`, we evaluate `sdf(fold(p))` where `fold()` wraps space into a repeating lattice.

For a simple 1D repetition with period `L`:
```rust
fn fold_1d(p: f32, L: f32) -> f32 {
    L * (p / L).round()  // Snap to nearest cell
}
```

In 3D, we can define a lattice with basis vectors and fold along 1, 2, or 3 axes:

```rust
struct SpaceFolding {
    lattice_basis: Mat3,      // Defines the repetition grid
    lattice_basis_inv: Mat3,  // For efficient transformation
    active_mask: u32,         // Which axes to fold (bitmask)
}

fn fold(p: Vec3, folding: SpaceFolding) -> Vec3 {
    // Transform to lattice space
    let lc = p * folding.lattice_basis_inv;
    
    // Round active coordinates to nearest cell
    let kx = if folding.active_mask & 1 { lc.x.round() } else { 0.0 };
    let ky = if folding.active_mask & 2 { lc.y.round() } else { 0.0 };
    let kz = if folding.active_mask & 4 { lc.z.round() } else { 0.0 };
    
    // Apply offset in world space
    p + Vec3(kx, ky, kz) * folding.lattice_basis
}
```

### Safety Thickness

One issue with space folding: if you're inside an object when folding occurs, you might "stick" by jumping to the wrong cell. I solve this with a minimum safety thickness:

```rust
min_half_thickness: 0.5 * min(lattice_u_length, lattice_v_length, lattice_w_length)
```

If the SDF returns negative (you're inside), I clamp the distance to push the ray out to this safe radius before the next fold. This prevents tunneling artifacts.

---

## Architecture: ECS Meets GPU Pipelines

The engine uses a fairly standard ECS (Entity-Component-System) architecture, but with a twist: **every component has a GPU mirror** that's kept in sync through dirty tracking.

### The Flow

```
┌─────────────┐
│  Rust ECS   │  Host-side scene state
│   World     │  Components in sparse sets
└──────┬──────┘
       │ dirty tracking
       ↓
┌─────────────┐
│ GPU Buffers │  Device-side compact arrays
│ GpuBuffer<T>│  Power-of-two growth, DtoD migration
└──────┬──────┘
       │ kernel parameters
       ↓
┌─────────────┐
│ CUDA Graph  │  Pre-recorded execution graph
│  Execution  │  Hot-swappable parameters per frame
└─────────────┘
```

### Component Synchronization

Each component type has:
- A `SparseSet<T>` on the host (entity index → component data)
- A `GpuBuffer<GpuT>` on the device (dense array of GPU structs)
- A `GpuIndexMap` (entity → GPU slot, with free list)

When a component changes:

1. The `SparseSet` marks it dirty
2. `update_scene()` detects dirty entities
3. The corresponding `GpuT` struct is rebuilt and uploaded to its slot
4. Dirty flags are cleared

For example, when a transform changes:

```rust
// Mark dirty (automatic on get_mut)
if let Some(transform) = world.transforms.get_mut(entity) {
    transform.position = new_pos;  // Marks dirty
}

// Later in update_scene():
for (_, entity_idx) in world.transforms.iter_dirty() {
    let entity = Entity { index: entity_idx, generation: ... };
    let transform = world.transforms.get(entity).unwrap();
    
    // Does this entity have an SDF? Update its GPU representation
    if let Some(sdf) = world.sdf_bases.get(entity) {
        let gpu_slot = world.sdf_gpu_indices.get_or_allocate_for(entity);
        let gpu_sdf = GpuSdfObjectBase {
            center: transform.position,
            u: transform.rotation * Vec3::X,
            v: transform.rotation * Vec3::Y,
            w: transform.rotation * Vec3::Z,
            // ... other fields
        };
        world.gpu_sdf_objects.push(gpu_slot, &gpu_sdf)?;
    }
}
```

This means:
- Only changed entities trigger GPU uploads
- Multiple component changes to the same entity coalesce into one update
- GPU memory layout stays compact (no holes from despawned entities)

### CUDA Graphs for Predictable Performance

Instead of launching kernels individually each frame, I build a **CUDA graph** once at startup:

```rust
let mut graph = cuda_context.create_cuda_graph()?;

// Add nodes
cuda_context.add_graph_kernel_node(
    &mut graph, 
    "generate_rays", 
    &params_generate_rays,
    grid_dim, 
    block_dim
)?;

cuda_context.add_graph_kernel_node(
    &mut graph,
    "raymarch",
    &params_raymarch,
    grid_dim,
    block_dim
)?;

// Link them
cuda_context.add_dependency(graph, "generate_rays", "raymarch")?;

// Instantiate for execution
let graph_exec = cuda_context.instantiate_graph(graph)?;
```

Each frame, I just launch the graph:

```rust
cuda_context.launch_graph(graph_exec)?;
```

When the scene changes and buffer pointers update, I hot-swap the kernel parameters without rebuilding:

```rust
if scene_updated {
    cam_ptr = world.gpu_cameras.ptr();
    sdf_ptr = world.gpu_sdf_objects.ptr();
    // ... update other pointers
    
    cuda_context.exec_kernel_node_set_params(
        graph_exec,
        "generate_rays",
        &params_generate_rays
    )?;
    
    cuda_context.exec_kernel_node_set_params(
        graph_exec,
        "raymarch",
        &params_raymarch  
    )?;
}
```

This keeps launch overhead low (~microseconds) and makes frame times predictable.

### Memory Management

`GpuBuffer<T>` handles device memory with:
- **Power-of-two growth** to minimize reallocations
- **Device-to-device migration** when resizing (no round-trip through host)
- **Stable indices** through the `GpuIndexMap` free list
- **In-place deactivation** by writing to an `active` flag offset

When an entity is despawned:

```rust
world.despawn(entity);

// Later in update_scene():
while let Some(entity_idx) = world.entities_to_remove_from_gpu.pop() {
    if let Some(slot) = world.sdf_gpu_indices.get(entity) {
        let active_offset = offset_of!(GpuSdfObjectBase, active);
        world.gpu_sdf_objects.deactivate(slot, active_offset)?;
        world.sdf_gpu_indices.free_for(entity);  // Return slot to free list
    }
}
```

The GPU slot is marked inactive (so kernels skip it) but the memory stays allocated for reuse. No compaction needed until the next rebuild.

---

## Textures and Materials

Materials can use solid colors or textures loaded from disk. The texture manager caches images and tracks reference counts:

```rust
pub struct TextureManager {
    cache: HashMap<PathBuf, Entry>,  // path → (handle, ref_count)
}

pub struct TextureHandle {
    d_ptr: CUdeviceptr,  // Device memory pointer
    width: u32,
    height: u32,
}
```

Loading a texture:

```rust
let texture = tex_mgr.load(Path::new("assets/texture.png"))?;
world.insert_material(entity, MaterialComponent {
    color: [1.0, 1.0, 1.0],
    texture: Some(texture),
    use_texture: true,
});
```

The manager uploads the raw RGB bytes to device memory once, then shares the handle across all entities using that texture.

### UV Mapping

On the GPU, I dispatch to different UV generation strategies:

- **Spherical mapping** for spheres: Standard lat/long parameterization
- **Triplanar mapping** for boxes and other shapes: Project texture from 3 axes and blend by normal

```cuda
__device__ Vec3 obj_mapping(
    const GpuMaterial& mat,
    const GpuSdfObjectBase& obj,
    Vec3 p,
    Vec3 center
) {
    if (obj.sdf_type == SDF_SPHERE) {
        // Spherical UV
        Vec3 dir = (p - center).normalize();
        Vec3 local_dir = world_to_local_rotation(dir, obj.u, obj.v, obj.w);
        float u = 0.5 + atan2(local_dir.z, local_dir.x) / (2*PI);
        float v = 0.5 - asin(local_dir.y) / PI;
        return sample_texture(mat, Vec2(u, v));
    } else {
        // Triplanar box mapping
        Vec3 normal = compute_normal(obj, p, center);
        return triplanar_sample(mat, obj, p, normal, center);
    }
}
```

Triplanar blending avoids distortion on non-spherical shapes by sampling the texture from three orthogonal planes and blending based on the surface normal.

---

## Rendering Pipeline Details

### Ray Generation with Depth of Field

The `generate_rays` kernel runs once per pixel to compute camera rays. For depth of field, I jitter the ray origin:

```cuda
if (cam.aperture > 0.0) {
    // Initialize RNG per pixel (once)
    if (needs_init) {
        curand_init(seed, pixel_index, 0, &cam.rand_states[i]);
    }
    
    // Sample a point on the lens
    Vec3 offset = random_in_unit_disk(&rand_state) * (cam.aperture * 0.5);
    origin = cam.position + cam.u * offset.x + cam.v * offset.y;
    
    // Aim at the focus plane
    Vec3 focus_point = cam.position + ray_dir * cam.focus_dist;
    direction = (focus_point - origin).normalize();
}
```

This creates a physically-based defocus blur effect. Aperture size controls blur amount, focus distance controls which plane is sharp.

### Lighting and Shading

Currently I use simple Lambertian shading with a directional light:

```cuda
Vec3 normal = evaluate_grad_sdf(obj, hit_point, center).normalize();
Vec3 light_dir = Vec3(0.5, -1.0, -0.6).normalize();
float diffuse = fmax(0.0, dot(normal, light_dir));
Vec3 shaded = base_color * (diffuse + 0.3);  // + ambient
```

The gradient of the SDF gives us the surface normal for free. For analytic SDFs (sphere, box, plane), these are exact. For others (cone, line), I fall back to a numerical gradient.

### Atmospheric Scattering

To add depth, I blend the shaded color toward a procedural sky based on ray march distance:

```cuda
// Horizon-biased fog density
float horizon_boost = 0.4 + 0.1 * (1.0 - clamp(ray_dir.y, 0, 1));
float fog = 1.0 - exp(-0.035 * horizon_boost * total_distance);

// Sky color from view direction
Vec3 sky = sky_color(ray_dir);

// Blend
Vec3 final_color = mix(shaded_color, sky, fog);
```

This creates atmospheric depth without explicit fog volumes. The horizon boost increases density when looking horizontally, matching real-world scattering.

---

## Performance Characteristics

### Frame Budget Breakdown (800×600, ~50 objects, 1 spp)

- **generate_rays**: ~0.1 ms (parallel per-pixel, trivial workload)
- **raymarch**: ~2-4 ms (most of the frame time)
  - SDF evaluation: ~60-70% of raymarch time
  - CSG tree traversal: ~20-25%  
  - Texturing/shading: ~10-15%
- **DtoH copy**: ~0.2 ms (async to pinned memory)
- **SDL present**: ~0.1 ms

Total: ~2.5-4.5 ms per frame → **220-400 FPS** for typical scenes.

### Scaling Observations

Ray marching performance is primarily bound by:

1. **Scene complexity**: More SDF objects = more evaluations per step
2. **CSG tree depth**: Deep trees mean more intermediate combines
3. **Maximum march distance**: Larger scenes need more steps to reach surfaces
4. **Screen resolution**: Linear scaling with pixel count

The CSG bounding spheres help significantly—if a ray is outside a tree's bound, I skip evaluating all its leaves. With space folding, I fold the bound center as well:

```cuda
Vec3 bound_center = tree.bound_center;
if (tree_folding_id != INVALID) {
    // Shift bound to nearest cell
    Vec3 k = fold_coords(p - bound_center, folding);
    bound_center = bound_center + k;
}

float dist_to_bound = length(p - bound_center);
if (dist_to_bound > current_min_dist + tree.bound_radius) {
    continue;  // Ray can't possibly be closer to this tree
}
```

This culls entire folded instances before evaluating their leaves.

---

## Future Directions

### BVH Acceleration

Currently I brute-force test every object and tree each step. For scenes with hundreds of objects, this becomes a bottleneck. A BVH (bounding volume hierarchy) would let me cull large chunks of the scene.

The challenge: building a BVH for implicit surfaces. Traditional BVHs use AABBs around triangles. With SDFs, I'd need conservative AABBs around each primitive, then rebuild the tree whenever transforms change.

### Multi-Light Support

Right now I have one hardcoded directional light. Supporting multiple point lights is straightforward—just loop and accumulate contributions. Soft shadows from area lights would be more interesting, probably using penumbra estimation during marching.

### PBR Materials

The current Lambertian shading is placeholder-tier. A proper PBR model (metallic/roughness workflow) would need:
- Microfacet BRDF evaluation
- Environment map sampling for indirect lighting  
- Multiple importance sampling for low-noise convergence

This would push me toward a path tracing approach rather than direct ray marching, which is a bigger architectural shift.

### Physics Integration

The ECS already has stubs for `RigidBody` and `Constraint` components. Integrating a physics solver (with SDF-based collision detection!) would make the engine interactive. SDFs give you contact normals and penetration depth for free—perfect for constraint resolution.

---

## Observations

### What Worked Well

**CUDA Graphs**: The up-front effort to set up graph execution paid off. Launch overhead went from ~50μs per kernel to <5μs per graph, and parameter hot-swapping is clean.

**Dirty Tracking**: Only syncing changed components keeps CPU-GPU bandwidth low. Most frames update nothing and just launch the same graph with the same parameters.

**Template-based CSG**: Unrolling the evaluation loop for each leaf count bucket gave a ~2× speedup over the naive dynamic approach. Registers matter.

**Space Folding**: This was surprisingly easy to implement and creates incredible visual complexity with near-zero cost. Highly recommend for SDF projects.

### What Was Hard

**CSG Tree Linearization**: Flattening arbitrary binary trees into arrays that preserve evaluation order and gradient signs took multiple rewrites. The in-order traversal was subtle—getting sibling pointers and parent backlinks right while avoiding cycles was error-prone.

**Pinned Memory Async Transfers**: CUDA's async memory model is powerful but tricky. I went through several iterations of using events and streams correctly to avoid stalls. The key insight: fence with an event, poll/sync on the event, *then* read the pinned buffer on the CPU side.

**Fold Safety Thickness**: Initial folding implementations had tunneling bugs where rays inside objects would jump to the wrong cell. The min_half_thickness hack works but feels fragile. A better solution might involve multi-step fold unwrapping.

---

## Try It Yourself

The repo includes a few demo scenes you can run:

```bash
# Build (requires CUDA toolkit, Rust, SDL2)
cargo build --release

# Run (default scene is the Menger sponge showcase)
cargo run --release
```

Controls:
- **Arrow keys**: Rotate camera (yaw/pitch)  
- **Space**: Move forward
- **Esc**: Exit


---

## References and Inspiration

**Inigo Quilez** — [iquilezles.org](https://iquilezles.org/)
- [Distance Functions Library](https://iquilezles.org/articles/distfunctions/) — Comprehensive collection of SDF primitives
- [Sphere Tracing](https://iquilezles.org/articles/raymarchingdf/) — The ray marching algorithm used in this engine
- [SDF Bounding Volumes](https://iquilezles.org/articles/sdfbounding/) — Acceleration techniques for complex scenes
- [Smooth Minimum](https://iquilezles.org/articles/smin/) — Advanced CSG blending operations

Inigo Quilez's work at Pixar on procedural modeling and his decades of shader demos (Shadertoy, Demoscene) have made techniques like ray marching SDFs practical and accessible. His articles are the definitive resource for this rendering approach.

