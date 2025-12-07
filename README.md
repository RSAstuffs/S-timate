<img width="794" height="580" alt="image" src="https://github.com/user-attachments/assets/fbc42034-185d-4b89-b0e3-2d3f50f34468" />
<img width="565" height="389" alt="image" src="https://github.com/user-attachments/assets/b32d50a3-7479-4f7c-8b02-0319f16c15ff" />


This is my pet project, S-timate.

S accuracy: **3.077419891515094e-303 PPM**

Which is: 3.08e-309 parts per billion
Or: 3.08e-312 parts per trillion

This is essentially ZERO error!

S differs from actual (p+q)/2 by only 1 unit.

It is by spamming signature verifications on a RSA public key that I'm able to derive that public key's S value with great accuracy.

This tool uses two methods, one is CPU based (Spectre protection and other timing-impeding protections must be disabled at a kernel level if using the CPU)

The other is RTL-SDR v5 based, my preference is RTL-SDR because it bypasses kernel-level noise.

After extracting my near perfect S, I use Z3 solver to finish off what needs correction.
