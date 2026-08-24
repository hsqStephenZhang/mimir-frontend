import mim
from mim._plugins.core import core
from mim._plugins.math import math

def _is_nat(world, t):
    return t == world.type_nat()

def _is_f32(world, t):
    try:
        return t == world.annex(math.F32.value)
    except:
        return False

def bin(op_name, lhs, rhs):
    world = None
    if isinstance(lhs, mim.Def):
        world = lhs.world()
    elif isinstance(rhs, mim.Def):
        world = rhs.world()
        
    if world is None:
        return NotImplemented
        
    def _to_def(val):
        if isinstance(val, mim.Def):
            return val
        if isinstance(val, int):
            return world.lit_nat(val)
        if isinstance(val, float):
            import struct
            bits = struct.unpack("<I", struct.pack("<f", float(val)))[0]
            return world.lit(world.annex(math.F32.value), bits)
        return NotImplemented

    lhs_def = _to_def(lhs)
    rhs_def = _to_def(rhs)
    
    if lhs_def is NotImplemented or rhs_def is NotImplemented:
        return NotImplemented

    t_lhs = lhs_def.type()
    t_rhs = rhs_def.type()

    if _is_nat(world, t_lhs) and _is_nat(world, t_rhs):
        if op_name == "add":
            callee = world.annex(core.nat.add.value)
        elif op_name == "sub":
            callee = world.annex(core.nat.sub.value)
        elif op_name == "mul":
            callee = world.annex(core.nat.mul.value)
        else:
            return NotImplemented
        return world.app(callee, world.tuple([lhs_def, rhs_def]))
    
    if _is_f32(world, t_lhs) and _is_f32(world, t_rhs):
        f32_config = world.annex(math.f32.value)
        # mode0 = world.lit_nat_0()
        mode80 = world.lit_nat(80)
        
        if op_name == "add":
            axm = world.annex(math.arith.add.value)
        elif op_name == "sub":
            axm = world.annex(math.arith.sub.value)
        elif op_name == "mul":
            axm = world.annex(math.arith.mul.value)
        elif op_name == "truediv":
            axm = world.annex(math.arith.div.value)
        else:
            return NotImplemented
            
        callee = world.app(world.app(axm, f32_config), mode80)
        return world.app(callee, [lhs_def, rhs_def])

    return NotImplemented

def add(lhs, rhs):
    return bin("add", lhs, rhs)

def sub(lhs, rhs):
    return bin("sub", lhs, rhs)

def mul(lhs, rhs):
    return bin("mul", lhs, rhs)

def div(lhs, rhs):
    return bin("truediv", lhs, rhs)
