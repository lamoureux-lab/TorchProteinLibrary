import torch
from torch.autograd import Function
from torch.autograd import Variable
from torch.nn.modules.module import Module
import _FullAtomModel
import math

import sys
import os


class Coords2TypedCoordsFunction(Function):
    """
    coordinates -> coordinated arranged in atom types function
    """

    @staticmethod
    def forward(ctx, input_coords_cpu, input_resnames, input_atomnames, num_atoms, num_atom_types):
        max_num_atoms = torch.max(num_atoms)

        if len(input_coords_cpu.size()) == 2:
            batch_size = input_coords_cpu.size(0)
            output_coords_cpu = torch.zeros(batch_size, num_atom_types, 3*max_num_atoms, dtype=input_coords_cpu.dtype)
            num_atoms_of_type = torch.zeros(batch_size, num_atom_types, dtype=torch.int)
            ctx.atom_indexes = torch.zeros(batch_size, num_atom_types, max_num_atoms, dtype=torch.int)

        else:
            raise ValueError('Coords2TypedCoordsFunction: ', 'Incorrect input size:', input_coords_cpu.size())

        _FullAtomModel.Coords2TypedCoords_forward(input_coords_cpu, input_resnames, input_atomnames, num_atoms,
                                                  output_coords_cpu, num_atoms_of_type, ctx.atom_indexes, num_atom_types.item())

        if math.isnan(output_coords_cpu.sum()):
            raise(Exception('Coords2TypedCoordsFunction: forward Nan'))

        ctx.save_for_backward(num_atoms_of_type, input_coords_cpu, num_atom_types)
        ctx.mark_non_differentiable(num_atoms_of_type)
        return output_coords_cpu, num_atoms_of_type

    @staticmethod
    def backward(ctx, grad_typed_coords_cpu, *kwargs):
        # print('Coords2TypedCoords backward')
        # ATTENTION! It passes non-contiguous tensor
        grad_typed_coords_cpu = grad_typed_coords_cpu.contiguous()
        num_atoms_of_type, input_coords_cpu, num_atom_types = ctx.saved_tensors

        if len(grad_typed_coords_cpu.size()) == 3:
            grad_coords_cpu = torch.zeros_like(input_coords_cpu)
        else:
            raise ValueError('Coords2TypedCoordsFunction: ', 'Incorrect input size:', input_angles_cpu.size())

        _FullAtomModel.Coords2TypedCoords_backward(	grad_typed_coords_cpu, grad_coords_cpu,
                                                    num_atoms_of_type,
                                                        ctx.atom_indexes, num_atom_types.item())

        if math.isnan(grad_coords_cpu.sum()):
            raise(Exception('Coords2TypedCoordsFunction: backward Nan'))

        return grad_coords_cpu, None, None, None, None


class Coords2TypedCoords(Module):
    def __init__(self, num_atom_types=11):
        super(Coords2TypedCoords, self).__init__()
        self.num_atom_types = torch.tensor([num_atom_types], dtype=torch.int)

    def forward(self, input_coords_cpu, input_resnames, input_atomnames, num_atoms):
        return Coords2TypedCoordsFunction.apply(input_coords_cpu, input_resnames, input_atomnames, num_atoms, self.num_atom_types)
