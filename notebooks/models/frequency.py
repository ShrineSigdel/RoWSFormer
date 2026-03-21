# class FrequencyEnhancement(nn.Module):
#     """
#     Uses Haar DWT for mathematically guaranteed frequency separation.
#     Watermark is encouraged into LL (low freq) subband which JPEG preserves.
#     Replaces approximate conv-based separation with exact frequency decomposition.
#     """
#     def __init__(self, channels):
#         super().__init__()

#         # Process LL subband (low freq — JPEG safe)
#         self.ll_refine = nn.Sequential(
#             nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
#             nn.Conv2d(channels, channels, 1),
#             nn.BatchNorm2d(channels), nn.GELU()
#         )

#         # Process HH/LH/HL subbands (high freq — JPEG destroys these)
#         self.hf_refine = nn.Sequential(
#             nn.Conv2d(channels * 3, channels, 1),  # merge LH+HL+HH → channels
#             nn.BatchNorm2d(channels), nn.GELU()
#         )

#         # Gate: learn how much of each subband to use
#         self.gate = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1), nn.Flatten(),
#             nn.Linear(channels, channels), nn.Sigmoid()
#         )

#         # Merge back to original channel dim
#         self.proj = nn.Conv2d(channels * 2, channels, 1)

#     def dwt(self, x):
#         """Haar DWT — mathematically exact frequency split."""
#         x00 = x[:, :, 0::2, 0::2]
#         x01 = x[:, :, 0::2, 1::2]
#         x10 = x[:, :, 1::2, 0::2]
#         x11 = x[:, :, 1::2, 1::2]
#         LL = (x00 + x01 + x10 + x11) / 4   # low freq  — JPEG preserves
#         LH = (x00 - x01 + x10 - x11) / 4   # horiz edges
#         HL = (x00 + x01 - x10 - x11) / 4   # vert edges
#         HH = (x00 - x01 - x10 + x11) / 4   # high freq — JPEG destroys
#         return LL, LH, HL, HH

#     def idwt(self, LL, LH, HL, HH):
#         """Haar IDWT — reconstruct from subbands."""
#         B, C, H, W = LL.shape
#         x = torch.zeros(B, C, H*2, W*2, device=LL.device, dtype=LL.dtype)
#         x[:, :, 0::2, 0::2] = LL + LH + HL + HH
#         x[:, :, 0::2, 1::2] = LL - LH + HL - HH
#         x[:, :, 1::2, 0::2] = LL + LH - HL - HH
#         x[:, :, 1::2, 1::2] = LL - LH - HL + HH
#         return x

#     def forward(self, x):
#         LL, LH, HL, HH = self.dwt(x)

#         # Refine low freq — this is where watermark should live
#         ll_out = self.ll_refine(LL)

#         # Merge and refine high freq subbands together
#         hf_cat = torch.cat([LH, HL, HH], dim=1)   # (B, C*3, H/2, W/2)
#         hf_out = self.hf_refine(hf_cat)

#         # Gate: bias toward low freq (JPEG robust)
#         g      = self.gate(ll_out).unsqueeze(-1).unsqueeze(-1)
#         merged = self.proj(torch.cat([ll_out * g, hf_out * (1 - g)], dim=1))

#         # Reconstruct to original spatial size via IDWT
#         # Use refined LL, keep LH/HL/HH mostly unchanged (don't disturb high freq unnecessarily)
#         out = self.idwt(merged, LH * 0.1, HL * 0.1, HH * 0.1)

#         return out + x   # residual — same as before