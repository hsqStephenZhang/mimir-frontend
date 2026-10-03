import mim
from mim._plugins.math import math

def make_world():
    driver = mim.Driver()
    driver.load_plugins(["math", "tensor", "core", "affine", "tuple"])
    return driver.world()

world = make_world()
F32 = world.annex(math.F32.value)

# input: [F32], output: F32
in_type = F32
out_type = F32

ret_cn_type = world.cn([out_type])
dom_with_ret = world.sigma([in_type, ret_cn_type])

lam = world.mut_con(dom_with_ret)
lam.set("my_cps_module")

args = lam.var()
x = args.proj(2, 0)
ret_cont = args.proj(2, 1)

# just return x
lam.app(True, ret_cont, [x])
lam.externalize()

with open("tests/draft_cps.mim", "w") as f:
    # dump might not return string directly or maybe it does, let's just write to file via write
    lam.write(mim.Dump.All, "tests/draft_cps.mim")

print("CPS module generated successfully.")
