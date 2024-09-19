"""
Copyright (c) 2022-2023, NVIDIA Corporation & affiliates. All rights reserved.


NVIDIA Source Code License for instant neural graphics primitives

@author Thomas Müller, NVIDIA

=======================================================================

1. Definitions

"Licensor" means any person or entity that distributes its Work.

"Software" means the original work of authorship made available under
this License.

"Work" means the Software and any additions to or derivative works of
the Software that are made available under this License.

The terms "reproduce," "reproduction," "derivative works," and
"distribution" have the meaning as provided under U.S. copyright law;
provided, however, that for the purposes of this License, derivative
works shall not include works that remain separable from, or merely
link (or bind by name) to the interfaces of, the Work.

Works, including the Software, are "made available" under this License
by including in or with the Work either (a) a copyright notice
referencing the applicability of this License to the Work, or (b) a
copy of this License.

2. License Grants

    2.1 Copyright Grant. Subject to the terms and conditions of this
    License, each Licensor grants to you a perpetual, worldwide,
    non-exclusive, royalty-free, copyright license to reproduce,
    prepare derivative works of, publicly display, publicly perform,
    sublicense and distribute its Work and any resulting derivative
    works in any form.

3. Limitations

    3.1 Redistribution. You may reproduce or distribute the Work only
    if (a) you do so under this License, (b) you include a complete
    copy of this License with your distribution, and (c) you retain
    without modification any copyright, patent, trademark, or
    attribution notices that are present in the Work.

    3.2 Derivative Works. You may specify that additional or different
    terms apply to the use, reproduction, and distribution of your
    derivative works of the Work ("Your Terms") only if (a) Your Terms
    provide that the use limitation in Section 3.3 applies to your
    derivative works, and (b) you identify the specific derivative
    works that are subject to Your Terms. Notwithstanding Your Terms,
    this License (including the redistribution requirements in Section
    3.1) will continue to apply to the Work itself.

    3.3 Use Limitation. The Work and any derivative works thereof only
    may be used or intended for use non-commercially. Notwithstanding
    the foregoing, NVIDIA and its affiliates may use the Work and any
    derivative works commercially. As used herein, "non-commercially"
    means for research or evaluation purposes only.

    3.4 Patent Claims. If you bring or threaten to bring a patent claim
    against any Licensor (including any claim, cross-claim or
    counterclaim in a lawsuit) to enforce any patents that you allege
    are infringed by any Work, then your rights under this License from
    such Licensor (including the grant in Section 2.1) will terminate
    immediately.

    3.5 Trademarks. This License does not grant any rights to use any
    Licensor’s or its affiliates’ names, logos, or trademarks, except
    as necessary to reproduce the notices described in this License.

    3.6 Termination. If you violate any term of this License, then your
    rights under this License (including the grant in Section 2.1) will
    terminate immediately.

4. Disclaimer of Warranty.

THE WORK IS PROVIDED "AS IS" WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WARRANTIES OR CONDITIONS OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE OR
NON-INFRINGEMENT. YOU BEAR THE RISK OF UNDERTAKING ANY ACTIVITIES UNDER
THIS LICENSE.

5. Limitation of Liability.

EXCEPT AS PROHIBITED BY APPLICABLE LAW, IN NO EVENT AND UNDER NO LEGAL
THEORY, WHETHER IN TORT (INCLUDING NEGLIGENCE), CONTRACT, OR OTHERWISE
SHALL ANY LICENSOR BE LIABLE TO YOU FOR DAMAGES, INCLUDING ANY DIRECT,
INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING OUT OF
OR RELATED TO THIS LICENSE, THE USE OR INABILITY TO USE THE WORK
(INCLUDING BUT NOT LIMITED TO LOSS OF GOODWILL, BUSINESS INTERRUPTION,
LOST PROFITS OR DATA, COMPUTER FAILURE OR MALFUNCTION, OR ANY OTHER
COMMERCIAL DAMAGES OR LOSSES), EVEN IF THE LICENSOR HAS BEEN ADVISED OF
THE POSSIBILITY OF SUCH DAMAGES.

=======================================================================
"""

import torch 

def sh_encoding(tensor, max_degree):
    x = tensor[:, 0]
    y = tensor[:, 1]
    z = tensor[:, 2]

    # Let compiler figure out how to sequence/reorder these calculations w.r.t. branches
    xy=x*y 
    xz=x*z 
    yz=y*z 
    x2=x*x 
    y2=y*y 
    z2=z*z
    x4=x2*x2 
    y4=y2*y2 
    z4=z2*z2
    x6=x4*x2 
    y6=y4*y2 
    z6=z4*z2

    data_out = torch.zeros(tensor.shape[0], 64, dtype=torch.float32, device=tensor.device)

    # SH polynomials generated using scripts/gen_sh.py based on the recurrence relations in appendix A1 of https:#www.ppsloan.org/publications/StupidSH36.pdf
    data_out[:, 0] = 0.28209479177387814                          # 1/(2*sqrt(pi))
    if (max_degree <= 1):
        return data_out[:, :1]

    data_out[:, 1] = -0.48860251190291987*y                               # -sqrt(3)*y/(2*sqrt(pi))
    data_out[:, 2] = 0.48860251190291987*z                                # sqrt(3)*z/(2*sqrt(pi))
    data_out[:, 3] = -0.48860251190291987*x                               # -sqrt(3)*x/(2*sqrt(pi))
    if (max_degree <= 2):
        return data_out[:, :4]
        
    data_out[:, 4] = 1.0925484305920792*xy                                # sqrt(15)*xy/(2*sqrt(pi))
    data_out[:, 5] = -1.0925484305920792*yz                               # -sqrt(15)*yz/(2*sqrt(pi))
    data_out[:, 6] = 0.94617469575755997*z2 - 0.31539156525251999                         # sqrt(5)*(3*z2 - 1)/(4*sqrt(pi))
    data_out[:, 7] = -1.0925484305920792*xz                               # -sqrt(15)*xz/(2*sqrt(pi))
    data_out[:, 8] = 0.54627421529603959*x2 - 0.54627421529603959*y2                              # sqrt(15)*(x2 - y2)/(4*sqrt(pi))
    if (max_degree <= 3):
        return data_out[:, :9]
        
    data_out[:, 9] = 0.59004358992664352*y*(-3.0*x2 + y2)                         # sqrt(70)*y*(-3*x2 + y2)/(8*sqrt(pi))
    data_out[:, 10] = 2.8906114426405538*xy*z                             # sqrt(105)*xy*z/(2*sqrt(pi))
    data_out[:, 11] = 0.45704579946446572*y*(1.0 - 5.0*z2)                                # sqrt(42)*y*(1 - 5*z2)/(8*sqrt(pi))
    data_out[:, 12] = 0.3731763325901154*z*(5.0*z2 - 3.0)                         # sqrt(7)*z*(5*z2 - 3)/(4*sqrt(pi))
    data_out[:, 13] = 0.45704579946446572*x*(1.0 - 5.0*z2)                                # sqrt(42)*x*(1 - 5*z2)/(8*sqrt(pi))
    data_out[:, 14] = 1.4453057213202769*z*(x2 - y2)                              # sqrt(105)*z*(x2 - y2)/(4*sqrt(pi))
    data_out[:, 15] = 0.59004358992664352*x*(-x2 + 3.0*y2)                                # sqrt(70)*x*(-x2 + 3*y2)/(8*sqrt(pi))
    if (max_degree <= 4):
        return data_out[:, :16]
        
    data_out[:, 16] = 2.5033429417967046*xy*(x2 - y2)                             # 3*sqrt(35)*xy*(x2 - y2)/(4*sqrt(pi))
    data_out[:, 17] = 1.7701307697799304*yz*(-3.0*x2 + y2)                                # 3*sqrt(70)*yz*(-3*x2 + y2)/(8*sqrt(pi))
    data_out[:, 18] = 0.94617469575756008*xy*(7.0*z2 - 1.0)                               # 3*sqrt(5)*xy*(7*z2 - 1)/(4*sqrt(pi))
    data_out[:, 19] = 0.66904654355728921*yz*(3.0 - 7.0*z2)                               # 3*sqrt(10)*yz*(3 - 7*z2)/(8*sqrt(pi))
    data_out[:, 20] = -3.1735664074561294*z2 + 3.7024941420321507*z4 + 0.31735664074561293                                # 3*(-30*z2 + 35*z4 + 3)/(16*sqrt(pi))
    data_out[:, 21] = 0.66904654355728921*xz*(3.0 - 7.0*z2)                               # 3*sqrt(10)*xz*(3 - 7*z2)/(8*sqrt(pi))
    data_out[:, 22] = 0.47308734787878004*(x2 - y2)*(7.0*z2 - 1.0)                                # 3*sqrt(5)*(x2 - y2)*(7*z2 - 1)/(8*sqrt(pi))
    data_out[:, 23] = 1.7701307697799304*xz*(-x2 + 3.0*y2)                                # 3*sqrt(70)*xz*(-x2 + 3*y2)/(8*sqrt(pi))
    data_out[:, 24] = -3.7550144126950569*x2*y2 + 0.62583573544917614*x4 + 0.62583573544917614*y4                         # 3*sqrt(35)*(-6*x2*y2 + x4 + y4)/(16*sqrt(pi))
    if (max_degree <= 5):
        return data_out[:, :25]
        
    data_out[:, 25] = 0.65638205684017015*y*(10.0*x2*y2 - 5.0*x4 - y4)                            # 3*sqrt(154)*y*(10*x2*y2 - 5*x4 - y4)/(32*sqrt(pi))
    data_out[:, 26] = 8.3026492595241645*xy*z*(x2 - y2)                           # 3*sqrt(385)*xy*z*(x2 - y2)/(4*sqrt(pi))
    data_out[:, 27] = -0.48923829943525038*y*(3.0*x2 - y2)*(9.0*z2 - 1.0)                         # -sqrt(770)*y*(3*x2 - y2)*(9*z2 - 1)/(32*sqrt(pi))
    data_out[:, 28] = 4.7935367849733241*xy*z*(3.0*z2 - 1.0)                              # sqrt(1155)*xy*z*(3*z2 - 1)/(4*sqrt(pi))
    data_out[:, 29] = 0.45294665119569694*y*(14.0*z2 - 21.0*z4 - 1.0)                             # sqrt(165)*y*(14*z2 - 21*z4 - 1)/(16*sqrt(pi))
    data_out[:, 30] = 0.1169503224534236*z*(-70.0*z2 + 63.0*z4 + 15.0)                            # sqrt(11)*z*(-70*z2 + 63*z4 + 15)/(16*sqrt(pi))
    data_out[:, 31] = 0.45294665119569694*x*(14.0*z2 - 21.0*z4 - 1.0)                             # sqrt(165)*x*(14*z2 - 21*z4 - 1)/(16*sqrt(pi))
    data_out[:, 32] = 2.3967683924866621*z*(x2 - y2)*(3.0*z2 - 1.0)                               # sqrt(1155)*z*(x2 - y2)*(3*z2 - 1)/(8*sqrt(pi))
    data_out[:, 33] = -0.48923829943525038*x*(x2 - 3.0*y2)*(9.0*z2 - 1.0)                         # -sqrt(770)*x*(x2 - 3*y2)*(9*z2 - 1)/(32*sqrt(pi))
    data_out[:, 34] = 2.0756623148810411*z*(-6.0*x2*y2 + x4 + y4)                         # 3*sqrt(385)*z*(-6*x2*y2 + x4 + y4)/(16*sqrt(pi))
    data_out[:, 35] = 0.65638205684017015*x*(10.0*x2*y2 - x4 - 5.0*y4)                            # 3*sqrt(154)*x*(10*x2*y2 - x4 - 5*y4)/(32*sqrt(pi))
    if (max_degree <= 6):
        return data_out[:, :36]
        
    data_out[:, 36] = 1.3663682103838286*xy*(-10.0*x2*y2 + 3.0*x4 + 3.0*y4)                               # sqrt(6006)*xy*(-10*x2*y2 + 3*x4 + 3*y4)/(32*sqrt(pi))
    data_out[:, 37] = 2.3666191622317521*yz*(10.0*x2*y2 - 5.0*x4 - y4)                            # 3*sqrt(2002)*yz*(10*x2*y2 - 5*x4 - y4)/(32*sqrt(pi))
    data_out[:, 38] = 2.0182596029148963*xy*(x2 - y2)*(11.0*z2 - 1.0)                             # 3*sqrt(91)*xy*(x2 - y2)*(11*z2 - 1)/(8*sqrt(pi))
    data_out[:, 39] = -0.92120525951492349*yz*(3.0*x2 - y2)*(11.0*z2 - 3.0)                               # -sqrt(2730)*yz*(3*x2 - y2)*(11*z2 - 3)/(32*sqrt(pi))
    data_out[:, 40] = 0.92120525951492349*xy*(-18.0*z2 + 33.0*z4 + 1.0)                           # sqrt(2730)*xy*(-18*z2 + 33*z4 + 1)/(32*sqrt(pi))
    data_out[:, 41] = 0.58262136251873131*yz*(30.0*z2 - 33.0*z4 - 5.0)                            # sqrt(273)*yz*(30*z2 - 33*z4 - 5)/(16*sqrt(pi))
    data_out[:, 42] = 6.6747662381009842*z2 - 20.024298714302954*z4 + 14.684485723822165*z6 - 0.31784601133814211                         # sqrt(13)*(105*z2 - 315*z4 + 231*z6 - 5)/(32*sqrt(pi))
    data_out[:, 43] = 0.58262136251873131*xz*(30.0*z2 - 33.0*z4 - 5.0)                            # sqrt(273)*xz*(30*z2 - 33*z4 - 5)/(16*sqrt(pi))
    data_out[:, 44] = 0.46060262975746175*(x2 - y2)*(11.0*z2*(3.0*z2 - 1.0) - 7.0*z2 + 1.0)                               # sqrt(2730)*(x2 - y2)*(11*z2*(3*z2 - 1) - 7*z2 + 1)/(64*sqrt(pi))
    data_out[:, 45] = -0.92120525951492349*xz*(x2 - 3.0*y2)*(11.0*z2 - 3.0)                               # -sqrt(2730)*xz*(x2 - 3*y2)*(11*z2 - 3)/(32*sqrt(pi))
    data_out[:, 46] = 0.50456490072872406*(11.0*z2 - 1.0)*(-6.0*x2*y2 + x4 + y4)                          # 3*sqrt(91)*(11*z2 - 1)*(-6*x2*y2 + x4 + y4)/(32*sqrt(pi))
    data_out[:, 47] = 2.3666191622317521*xz*(10.0*x2*y2 - x4 - 5.0*y4)                            # 3*sqrt(2002)*xz*(10*x2*y2 - x4 - 5*y4)/(32*sqrt(pi))
    data_out[:, 48] = 10.247761577878714*x2*y4 - 10.247761577878714*x4*y2 + 0.6831841051919143*x6 - 0.6831841051919143*y6                         # sqrt(6006)*(15*x2*y4 - 15*x4*y2 + x6 - y6)/(64*sqrt(pi))
    if (max_degree <= 7):
        return data_out[:, :49]
        
    data_out[:, 49] = 0.70716273252459627*y*(-21.0*x2*y4 + 35.0*x4*y2 - 7.0*x6 + y6)                              # 3*sqrt(715)*y*(-21*x2*y4 + 35*x4*y2 - 7*x6 + y6)/(64*sqrt(pi))
    data_out[:, 50] = 5.2919213236038001*xy*z*(-10.0*x2*y2 + 3.0*x4 + 3.0*y4)                             # 3*sqrt(10010)*xy*z*(-10*x2*y2 + 3*x4 + 3*y4)/(32*sqrt(pi))
    data_out[:, 51] = -0.51891557872026028*y*(13.0*z2 - 1.0)*(-10.0*x2*y2 + 5.0*x4 + y4)                          # -3*sqrt(385)*y*(13*z2 - 1)*(-10*x2*y2 + 5*x4 + y4)/(64*sqrt(pi))
    data_out[:, 52] = 4.1513246297620823*xy*z*(x2 - y2)*(13.0*z2 - 3.0)                           # 3*sqrt(385)*xy*z*(x2 - y2)*(13*z2 - 3)/(8*sqrt(pi))
    data_out[:, 53] = -0.15645893386229404*y*(3.0*x2 - y2)*(13.0*z2*(11.0*z2 - 3.0) - 27.0*z2 + 3.0)                              # -3*sqrt(35)*y*(3*x2 - y2)*(13*z2*(11*z2 - 3) - 27*z2 + 3)/(64*sqrt(pi))
    data_out[:, 54] = 0.44253269244498261*xy*z*(-110.0*z2 + 143.0*z4 + 15.0)                              # 3*sqrt(70)*xy*z*(-110*z2 + 143*z4 + 15)/(32*sqrt(pi))
    data_out[:, 55] = 0.090331607582517306*y*(-135.0*z2 + 495.0*z4 - 429.0*z6 + 5.0)                              # sqrt(105)*y*(-135*z2 + 495*z4 - 429*z6 + 5)/(64*sqrt(pi))
    data_out[:, 56] = 0.068284276912004949*z*(315.0*z2 - 693.0*z4 + 429.0*z6 - 35.0)                              # sqrt(15)*z*(315*z2 - 693*z4 + 429*z6 - 35)/(32*sqrt(pi))
    data_out[:, 57] = 0.090331607582517306*x*(-135.0*z2 + 495.0*z4 - 429.0*z6 + 5.0)                              # sqrt(105)*x*(-135*z2 + 495*z4 - 429*z6 + 5)/(64*sqrt(pi))
    data_out[:, 58] = 0.07375544874083044*z*(x2 - y2)*(143.0*z2*(3.0*z2 - 1.0) - 187.0*z2 + 45.0)                         # sqrt(70)*z*(x2 - y2)*(143*z2*(3*z2 - 1) - 187*z2 + 45)/(64*sqrt(pi))
    data_out[:, 59] = -0.15645893386229404*x*(x2 - 3.0*y2)*(13.0*z2*(11.0*z2 - 3.0) - 27.0*z2 + 3.0)                              # -3*sqrt(35)*x*(x2 - 3*y2)*(13*z2*(11*z2 - 3) - 27*z2 + 3)/(64*sqrt(pi))
    data_out[:, 60] = 1.0378311574405206*z*(13.0*z2 - 3.0)*(-6.0*x2*y2 + x4 + y4)                         # 3*sqrt(385)*z*(13*z2 - 3)*(-6*x2*y2 + x4 + y4)/(32*sqrt(pi))
    data_out[:, 61] = -0.51891557872026028*x*(13.0*z2 - 1.0)*(-10.0*x2*y2 + x4 + 5.0*y4)                          # -3*sqrt(385)*x*(13*z2 - 1)*(-10*x2*y2 + x4 + 5*y4)/(64*sqrt(pi))
    data_out[:, 62] = 2.6459606618019*z*(15.0*x2*y4 - 15.0*x4*y2 + x6 - y6)                               # 3*sqrt(10010)*z*(15*x2*y4 - 15*x4*y2 + x6 - y6)/(64*sqrt(pi))
    data_out[:, 63] = 0.70716273252459627*x*(-35.0*x2*y4 + 21.0*x4*y2 - x6 + 7.0*y6)                              # 3*sqrt(715)*x*(-35*x2*y4 + 21*x4*y2 - x6 + 7*y6)/(64*sqrt(pi))

    return data_out[:, :64]


NUM_PARAMS_FOR_ENCODING = [None, 1, 4, 9, 16, 25, 36, 49, 64]