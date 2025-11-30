---
layout: page
title: Projects
permalink: /personal_projects/
---

# Personal Projects

- **[Rust + CUDA Ray-Marching Engine](/2025/06/12/Raymarching-Engine.html)**<br>
Real-time ray marching engine that renders procedural geometry entirely on the GPU. Combines Rust for safe host-side code with custom CUDA kernels for parallel evaluation of signed distance functions.<br><br>

- **Evolutionary Image Generation with GPU-Accelerated Particle Placement**<br>
[GitHub](https://github.com/EtiNL/Evolution_generated_images)<br>
Developed an evolutionary algorithm that recreates artwork through iterative particle placement, using CUDA-accelerated rendering to generate thousands of colored circles. <br>
The algorithm employs depth-map-based candidate selection, bisection search for optimal radius determination, and fitness-based particle selection to minimize pixel-wise error between target and generated images. <br>
<div class="video-wrap">
  <iframe
    src="https://www.youtube-nocookie.com/embed/b_EqgPfkqOU"
    title="Generation process video"
    frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
</div>

<br><br>

- **AI Model for Agent Selection and Scoring - CLGP**<br>
[GitHub](https://github.com/EtiNL/Ebiose_CLGP) | [Presentation](/assets/pdf/Presentation_Ebiose.pdf)<br>
I developed a Contrastive Language Graph Pair (CLGP) model for Ebiose AI, to efficiently match user prompts with optimal AI agents and evaluate agent architectures before deployment. <br>
The model combines BERT text encoders with Graph Neural Networks to compute compatibility scores via cosine similarity, significantly reducing computational cost.<br>
Applied to mathematical reasoning tasks on the GSM8k benchmark, this lightweight solution enables resource-efficient agent selection in production and faster agent generation workflows.