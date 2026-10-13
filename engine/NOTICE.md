# Third-party notice

`engine/regional.py` is derived from
[ComfyUI-Krea2-Regional](https://github.com/januspluto/ComfyUI-Krea2-Regional)
by januspluto, commit `307081f2b954d9f5e683dadafdb53ca971ebdbfd`, released under
the MIT license. The original license text is kept verbatim in
[`LICENSE-krea2-regional`](LICENSE-krea2-regional).

What was kept: LoRA state-dict parsing (LoRA and LoKr), the token-masked LoRA
class swap, the joint attention mask and the diffusion-model wrapper.

What was changed: node classes, adaptive masks and Region Lock were removed;
wrapper and bundle keys were renamed so the engine never collides with an
installed copy of the upstream extension; prompt chunks may carry NegPiP
sidecar tokens, and segments are laid out on the stripped sequence.

The upstream builder, caption helpers, canvas editor, captioner and detailer
are not used.
