from spy.conversions import *

import os

import pymatgen
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.symmetry.bandstructure import HighSymmKpath
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


from mpinterfaces.interface import Interface

import re

import scipy
import numpy as np

import matplotlib.pyplot as plt


class Structure(pymatgen.Structure):
    
    """
    Same as a Pymatgen Structure object, but able to write & read
    SPHInX structure files. **All distances are stored as Angstroms
    and only converted to Bohr when writing the file**
    """

    def to_sx_file(self, filename="structure.sx", selective_dynamics=[]):
        """
        Write a SPHInX C++-style structure file.

        `selective_dynamics` (list): list of 3D tuples specifying
            ionic degrees of freedom for each atom, e.g.
            [(True, True, True), (False, False, True)... ].
            Defaults to all (True, True, True).

        `filename` (str): output filename.
        """

        a, b, c = self.lattice.matrix * A_TO_BOHR
        coords = {
            t: [[x*A_TO_BOHR for x in s.coords]
                for s in self.sites if s.specie == t
            ] for t in self.species
        }

        if len(selective_dynamics) == 0:
            selective_dynamics = [
                (True, True, True) for i in range(len(self.sites))]

        with open(filename, "w") as sx:
            sx.write("structure {\n")
            sx.write(f"   cell = [{a}, {b}, {c}];\n")
            n = 0
            for elt in coords:
                sx.write(f"   species {{\n")
                sx.write(f"{TAB2}element=\"{elt}\";\n")
                for coord in coords[elt]:
                    sx.write(f"{TAB2}atom {{ coords = {coord}; ")
                    if selective_dynamics[n][0]:
                        sx.write("movableX; ")
                    if selective_dynamics[n][1]:
                        sx.write("movableY; ")
                    if selective_dynamics[n][2]:
                        sx.write("movableZ; ")
                    sx.write("}\n")
                    n += 1
                sx.write(f"   \n")
            sx.write("}\n")


    def from_sx_file(filename="structure.sx"):
        sx = open(filename).read()

        open_squares = [m.start() for m in re.finditer("\[", sx)]
        closed_squares = [m.start() for m in re.finditer("\]", sx)]
        open_curlies = [m.start() for m in re.finditer("\{", sx)]
        closed_curlies = [m.start() for m in re.finditer("\}", sx)]
        semicolons = [m.start() for m in re.finditer(";", sx)]
        species = [m.end() for m in re.finditer("species", sx)]
        elements = [m.end() for m in re.finditer("element", sx)]
        atoms = [m.end() for m in re.finditer("atom", sx)]
        coords = [m.end() for m in re.finditer("coords", sx)]

        a = np.array([float(x) for x in
            sx[open_squares[1]+1:closed_squares[0]].split(",")])
        b = np.array([float(x) for x in
            sx[open_squares[2]+1:closed_squares[1]].split(",")])
        c = np.array([float(x.replace(";", "")) for x in
            sx[open_squares[3]+1:closed_squares[2]].split(",")])

        lattice = np.divide(np.array([a, b, c]), A_TO_BOHR)

        ei = 0
        ai = 0
        species_names = []
        coordinates = []
        for si in range(len(species)):
            elt_start = elements[ei]
            elt_end = min([s for s in semicolons if s > elt_start])
            element = sx[elt_start:elt_end].strip().replace(
                "\"","").replace("=", "")
            ei += 1

            if si == len(species)-1:
                n_atoms = len([a for a in range(len(atoms)) if
                    atoms[a] > species[si]])
            else:
                n_atoms = len([a for a in range(len(atoms)) if
                    species[si+1] > atoms[a] > species[si]])
            sai = 0
            while (sai < n_atoms):
                c_start = min([oi for oi in open_squares if oi > coords[ai]])+1
                c_end = min([ci for ci in closed_squares if ci > coords[ai]])
                coord = np.array([float(x) for x in
                    sx[c_start:c_end].split(",")])/A_TO_BOHR
                species_names.append(element)
                coordinates.append(coord)
                ai += 1
                sai += 1

        return Structure(lattice=lattice, species=species_names,
                         coords=coordinates, coords_are_cartesian=True)


def make_slab(basis, hkl, min_thickness, min_vacuum,
              selective_dynamics=[], supercell=(1,1,1)):
    iface = Interface(basis, hkl=hkl,
                  min_thick=min_thickness, min_vac=min_vacuum,
                  primitive=False, from_ase=True)
    iface.create_interface()
    iface.sort()

    if len(selective_dynamics) == 0:
        selective_dynamics = [(True, True, True) for i in iface.sites]
    iface_poscar = Poscar(iface, selective_dynamics=selective_dynamics)
    slab = Structure.from_str(iface_poscar.get_string(), fmt="poscar")
    sga = SpacegroupAnalyzer(slab)

    # This is not *necessary*, but often gives a
    # much prettier unit cell whose periodicity
    # is easier to understand.
    slab = sga.get_primitive_standard_structure()
    slab.make_supercell(supercell)
    return Structure(lattice=slab.lattice, coords=slab.cart_coords,
        species=slab.species, coords_are_cartesian=True)


def write_input(structure, ecut=500, kpoints=[10,10,1], xc="PBE", charge=0,
                charge_z=0, vdw=None, dE=1e-6, dF=1e-5, n_steps=150):

    ecut *= EV_TO_RY
    charge_z *= A_TO_BOHR
    dE *= EV_TO_HA
    dF *= EV_TO_HA / A_TO_BOHR
    inp = open("input.sx", "w")
    inp.write("format paw;\ninclude <parameters.sx>;\npawPot {\n")
    for element in [s for s in structure.composition.as_dict()]:
        inp.write("   species {\n")
        inp.write(f"{TAB2}name = \"{element}\";\n")
        inp.write(f"{TAB2}potType = \"VASP\";\n")
        inp.write(f"{TAB2}element = \"{element}\";\n")
        inp.write(f"{TAB2}potential = <PAWpot/{element}-{xc.lower()}.vasppaw>;\
            \n   }}\n")
    inp.write("}\n\n")
    inp.write("include \"structure.sx\";\n\n")
    inp.write(f"basis {{\n   eCut={ecut};\
        \n   kPoint {{ coords=[1/2,1/2,1/4]; weight=1; relative;}}\
        \n   folding={kpoints};\n}}\n\n")
    inp.write("PAWHamiltonian  {\n")
    inp.write("   nEmptyStates = 20;\n")
    inp.write("   ekt = 0.1;\n")
    inp.write(f"   xc = {xc};\n")
    if charge is not None:
        inp.write("   dipoleCorrection;\n")
        inp.write(f"   nExcessElectrons = -{charge};\n")
    if vdw is not None:
        inp.write(f"   vdwCorrection {{method = \"{vdw}\";}}\n")
    inp.write("}\n\n")
    inp.write("initialGuess  {\n")
    inp.write("   waves { lcao { maxSteps=1; rhoMixing = 0.; }; pawBasis;}\n")
    inp.write(f"   rho {{ atomicOrbitals; charged {{charge = {charge};\
        z={charge_z};}} }}\n")
    inp.write("}\n\n")
    inp.write("main  {\n   linQN {\n")
    inp.write("{TAB2}dEnergy = {dE};\n")
    inp.write("{TAB2}dF = {dF};\n")
    inp.write(f"{TAB2}maxSteps={n_steps};\n")
    inp.write(f"{TAB2}bornOppenheimer {{\n{TAB2}   scfDiag {{\
        \n{TAB2*2}rhoMixing= 0.5;\n")
    inp.write(f"{TAB2*2}blockCCG {{ blockSize=64; }}\
        \n{TAB2*2}dEnergy={dE};\n{TAB2*2}maxSteps={n_steps};\n")
    inp.write("{TAB2}   \n{TAB2}\n   }\n}")


def write_runjob(name, n_tasks, time):
    inp = open("runjob", "w")
    if n_tasks > 20:
        partition = "p.cmfe"
    else:
        partition = "s.cmfe"
    inp.write(f"#!/bin/bash\
        \n#SBATCH --partition={partition}\
        \n#SBATCH --ntasks={n_tasks}\
        \n#SBATCH --time={time}\
        \n#SBATCH --output=err.out\
        \n#SBATCH --job-name={name}\
        \n#SBATCH --get-user-env=L\n")
    inp.write(f"module load gcc impi\
        \nsrun -n {n_tasks} /u/mashton/software/sphinx/bin/sphinx\
        > sphinx.log")


def get_high_symmetry_kpoints_group(structure, n_points=20, symprec=0.01):
    kpts = HighSymmKpath(structure, symprec)
    warning = False
    while kpts.kpath is None:
        warning = True
        symprec *= 2
        kpts = HighSymmKpath(structure, symprec)
    if warning:
        print(f"Found first kpath using symprec={symprec}")

    start = list(kpts.kpath["kpoints"][kpts.kpath["path"][0][0]])
    label = kpts.kpath["path"][0][0].replace("\\Gamma", "\\xG")

    kpts_group = ""
    kpts_group += "kPoints {\n   relative;\n"
    kpts_group += f"   from {{ coords={start}; label=\"{label}\"; }}\n"
    for ki in range(1, len(kpts.kpath["path"][0])):
        coords = list(kpts.kpath["kpoints"][kpts.kpath["path"][0][ki]])
        label = kpts.kpath["path"][0][ki].replace("\\Gamma", "\\xG")
        kpts_group += f"   to {{ coords={coords}; label=\"{label}\"; nPoints={n_points}; }}\n"
    for path in kpts.kpath["path"][1:]:
        coords = list(kpts.kpath["kpoints"][path[0]])
        label = path[0].replace("\\Gamma", "\\xG")
        kpts_group += f"   to {{ coords={coords}; label=\"{label}\"; nPoints=0; }}\n"
        for ki in range(1, len(path)):
            coords = list(kpts.kpath["kpoints"][path[ki]])
            label = path[ki].replace("\\Gamma", "\\xG")
            kpts_group += f"   to {{ coords={coords}; label=\{label}\"; nPoints={n_points}; }}\n"
    kpts_group += "}"
    return kpts_group

def read_kpoints(filename="sphinx.log"):

    sx_lines = open(filename).readlines()
    kpoints = []
    for i in range(len(sx_lines)):
        if "Symmetrized k-points" in sx_lines[i]:
            n = i+2
            while "+-------------" not in sx_lines[n]:
                kpoints.append(sx_lines[n].split())
                n += 1
            break
    coords, labels = [], []
    for k in kpoints:
        coords.append(np.array([float(c) for c in k[2:5]]))
        if len(k) == 9:
            labels.append(k[8])
        else:
            labels.append(None)
    return coords, labels


def plot_band_structure(output_filename="bands.pdf", spins=[0,1],
                        ylim=[-5, 5], figsize=(12, 8), logfile="sphinx.log"):

    ax = plt.figure(figsize=figsize).gca()
    eps_lines = []

    log_lines = open(logfile).readlines()
    fermi_line = [l for l in log_lines if "Fermi energy" in l][-1]
    fermi_energy = float(fermi_line.split()[-2])

    coords, labels = read_kpoints()

    if os.path.isfile("eps.dat"):
        eps_lines.append(open("eps.dat").readlines()[2:])
        eps_lines.append([])
    elif os.path.isfile("eps.0.dat"):
        eps_lines.append(open("eps.0.dat").readlines()[2:])
        eps_lines.append(open("eps.1.dat").readlines()[2:])
    for iS in spins:
        bands = []
        for line in eps_lines[iS]:
            sl = line.split()
            eigenvalues = [float(e)-fermi_energy for e in sl[1:]]
            for i,e in enumerate(eigenvalues):
                if i >= len(bands):
                    bands.append([])
                bands[i].append(e)

        segment_lengths, kpt_distances = [], [0]
        hs_coords, hs_labels = [], []
        if labels[0] is not None:
            hs_coords.append(0)
            hs_labels.append(labels[0])
        for ik in range(1, len(coords)):
            if labels[ik] is not None:
                hs_coords.append(distance)
                hs_labels.append(labels[ik].replace("=","|"))
                distance = kpt_distances[ik-1]
            else:
                distance = np.linalg.norm(coords[ik] - coords[ik-1]) + kpt_distances[ik-1]
            kpt_distances.append(distance)

        for band in bands:
            ax.plot(kpt_distances, band, color=plt.cm.rainbow(iS+0.2))

        for i in range(len(hs_coords)):
            ax.plot([hs_coords[i], hs_coords[i]], ylim, "k--")

    ax.set_xlim(0, max(kpt_distances))
    ax.set_xticks(hs_coords)
    ax.set_xticklabels(hs_labels)
    ax.set_ylim(ylim)

    plt.savefig(output_filename)


def get_hirshfeld_charges(filename="hirshfeld.sx", force_overwrite=False):
    if force_overwrite or not os.path.isfile(filename):
        os.system(f"~/software/sphinx/bin/sxpawatomvolume > {filename}")
    lines = open(filename).readlines()
    hirshfeld_line = [line for line in lines if "Hirshfeld" in line][0]
    hirshfeld_charges = [float(c) for c in
        hirshfeld_line.split("[")[1].split("]")[0].split(",")]
    return hirshfeld_charges


def create_charged_structure_file(filename="CONTCAR", convert_from_sphinx=False):
    if convert_from_sphinx or not os.path.isfile(filename):
        if os.path.isfile("relaxedStr.sx"):
            os.system(f"/u/mashton/software/sphinx/bin/sx2poscar -i relaxedStr.sx -o {filename}")
        else: 
            raise(f"ERROR: No relaxedStr.sx found in {os.getcwd()}")

    charges = get_hirshfeld_charges()

    pipeline = import_file(filename)
    pipeline.compute()
    data_collection = pipeline.output

    charge = ParticleProperty.create_user("Hirshfeld Charge", "float", data_collection.number_of_particles)
    for i in range(data_collection.number_of_particles):
        charge.marray[i] = charges[i]

    modifier = ColorCodingModifier(
        particle_property = "Hirshfeld Charge",
        gradient = ColorCodingModifier.Viridis()
    )

    data_collection.add(charge)

    pipeline.source = data_collection
    pipeline.modifiers.append(modifier)

    export_file(
        pipeline, "charged_structure.xyz", "xyz",
        columns=["Particle Identifier", "Particle Type", "Position.X",
                 "Position.Y", "Position.Z", "Hirshfeld Charge"]
    )
