import os

from pymatgen import Structure
from pymatgen.symmetry.bandstructure import HighSymmKpath

import re

import scipy
import numpy as np

import matplotlib.pyplot as plt


BOHR_TO_A = np.divide(
    scipy.constants.physical_constants["Bohr radius"][0],
    scipy.constants.physical_constants["Angstrom star"][0]
    )


def pymatgen_to_sx(structure, selective_dynamics=None, output_filename="structure.sx"):
    a = [x*1.88973 for x in structure.lattice.matrix[0]]
    b = [x*1.88973 for x in structure.lattice.matrix[1]]
    c = [x*1.88973 for x in structure.lattice.matrix[2]]
    coords = {
        t: [
            [x*1.88973 for x in s.coords] for s in structure.sites if s.specie == t
        ] for t in structure.species
    }

    if not selective_dynamics:
        selective_dynamics = [(True, True, True) for i in range(len(coords))]

    with open(output_filename, "w") as sx:
        sx.write("structure {\n")
        sx.write("    cell = [%s, %s, %s];\n" % (a, b, c))
        n = 0
        for elt in coords:
            sx.write("    species {\n")
            sx.write("        element=\"%s\";\n" % elt)
            for c in coords[elt]:
                sx.write("        atom { coords = %s; " % c)
                if selective_dynamics[n][0]:
                    sx.write("movableX; ")
                if selective_dynamics[n][1]:
                    sx.write("movableY; ")
                if selective_dynamics[n][2]:
                    sx.write("movableZ; ")
                sx.write("}\n")
                n += 1
            sx.write("    }\n")
        sx.write("}\n")


def sx_to_pymatgen(sx_filename="structure.sx"):
    sx = open(sx_filename).read()

    open_squares = [m.start() for m in re.finditer("\[", sx)]
    closed_squares = [m.start() for m in re.finditer("\]", sx)]
    open_curlies = [m.start() for m in re.finditer("\{", sx)]
    closed_curlies = [m.start() for m in re.finditer("\}", sx)]
    semicolons = [m.start() for m in re.finditer(";", sx)]
    species = [m.end() for m in re.finditer("species", sx)]
    elements = [m.end() for m in re.finditer("element", sx)]
    atoms = [m.end() for m in re.finditer("atom", sx)]
    coords = [m.end() for m in re.finditer("coords", sx)]

    a = np.array([float(x) for x in sx[open_squares[1]+1:closed_squares[0]].split(",")])*BOHR_TO_A
    b = np.array([float(x) for x in sx[open_squares[2]+1:closed_squares[1]].split(",")])*BOHR_TO_A
    c = np.array([float(x.replace(";", "")) for x in sx[open_squares[3]+1:closed_squares[2]].split(",")])*BOHR_TO_A

    lattice = np.array([a, b, c])

    ei = 0
    ai = 0
    species_names = []
    coordinates = []
    for si in range(len(species)):
        elt_start = elements[ei]
        elt_end = min([s for s in semicolons if s > elt_start])
        element = sx[elt_start:elt_end].strip().replace("\"", "").replace("=", "")
        ei += 1

        if si == len(species)-1:
            n_atoms = len([a for a in range(len(atoms)) if atoms[a] > species[si]])
        else:
            n_atoms = len([a for a in range(len(atoms)) if species[si+1] > atoms[a] > species[si]])
        sai = 0
        while (sai < n_atoms):
            c_start = min([oi for oi in open_squares if oi > coords[ai]])+1
            c_end = min([ci for ci in closed_squares if ci > coords[ai]])
            coord = np.array([float(x) for x in sx[c_start:c_end].split(",")])*BOHR_TO_A
            species_names.append(element)
            coordinates.append(coord)
            ai += 1
            sai += 1

    return Structure(lattice=lattice, species=species_names, coords=coordinates, coords_are_cartesian=True)


def write_input(structure, ecut=40, kpoints=[10,10,1], charge=0, charge_z=0, dE=1e-6, dF=1e-5, n_steps=150):
    inp = open("input.sx", "w")
    inp.write("format paw;\ninclude <parameters.sx>;\npawPot {\n")
    for element in [s for s in structure.composition.as_dict()]:
        inp.write("   species {\n")
        inp.write("      name           = \"%s\";\n" % element)
        inp.write("      potType        = \"VASP\";\n")
        inp.write("      element        = \"%s\";\n" % element)
        inp.write("      potential      = <PAWpot/%s-pbe.vasppaw>;\n   }\n" % element)
    inp.write("}\n\n")
    inp.write("include \"structure.sx\";\n\n")
    inp.write("basis {\n   eCut=%s;\n   kPoint { coords=[1/2,1/2,1/4]; weight=1; relative;}\n   folding=%s;\n}\n\n" % (ecut, kpoints))
    inp.write("PAWHamiltonian  {\n")
    inp.write("   nEmptyStates = 20;\n")
    inp.write("   ekt = 0.1;\n")
    inp.write("   xc         = PBE;\n")
    if charge is not None:
        inp.write("   dipoleCorrection;\n")
        inp.write("   nExcessElectrons= %s;\n" % -charge)
    inp.write("   vdwCorrection {method=\"D2\";}\n")
    inp.write("}\n\n")
    inp.write("initialGuess  {\n")
    inp.write("   waves { lcao { maxSteps=1; rhoMixing = 0.; }; pawBasis;}\n")
    inp.write("   rho { atomicOrbitals; charged {charge = %s; z=%s;} }\n" % (charge, charge_z))
    inp.write("}\n\n")
    inp.write("main  {\n   linQN {\n")
    inp.write("      dEnergy = %s;\n" % dE)
    inp.write("      dF = %s;\n" % dF)
    inp.write("      maxSteps=%s;\n" % n_steps)
    inp.write("      bornOppenheimer {\n         scfDiag {\n            rhoMixing= 0.5;\n")
    inp.write("            blockCCG { blockSize=64; }\n            dEnergy  = %s;\n            maxSteps = 150;\n" % dE)
    inp.write("         }\n      }\n   }\n}")


def write_runjob(name, n_tasks):
    inp = open("runjob", "w")
    if n_tasks > 20:
        partition = "p.cmfe"
    else:
        partition = "s.cmfe"
    inp.write("#!/bin/bash\n#SBATCH --partition=%s\n#SBATCH --ntasks=%s\n#SBATCH --time=5760\n#SBATCH --output=time.out\n#SBATCH --job-name=%s\n#SBATCH --get-user-env=L\n" % (partition, n_tasks, name))
    inp.write("module load gcc impi\nsrun -n %s /u/mashton/software/sphinx/bin/sphinx > sphinx.log" % n_tasks)


def get_high_symmetry_kpoints_group(sx_filename="structure.sx", npoints=20, symprec=0.01):
    s = sx_to_pymatgen(sx_filename)
    kpts = HighSymmKpath(s, symprec)
    warning = False
    while kpts.kpath is None:
        warning = True
        symprec *= 2
        kpts = HighSymmKpath(s, symprec)
    if warning:
        print("Could not identify kpath using symmetry precision below ", symprec)

    kpts_group = ""
    TAB = "    "
    kpts_group += "kPoints {\n" + TAB + "relative;\n"
    kpts_group += TAB + "from { coords=%s; label=\"%s\"; }\n" % (list(kpts.kpath["kpoints"][kpts.kpath["path"][0][0]]), kpts.kpath["path"][0][0].replace("\\Gamma", "\\xG"))
    for ki in range(1, len(kpts.kpath["path"][0])):
        kpts_group += TAB + "to { coords=%s; label=\"%s\"; nPoints=%s; }\n" % (list(kpts.kpath["kpoints"][kpts.kpath["path"][0][ki]]), kpts.kpath["path"][0][ki].replace("\\Gamma", "\\xG"), npoints)
    for path in kpts.kpath["path"][1:]:
        kpts_group += TAB + "to { coords=%s; label=\"%s\"; nPoints=0; }\n" % (list(kpts.kpath["kpoints"][path[0]]), path[0].replace("\\Gamma", "\\xG"))
        for ki in range(1, len(path)):
            kpts_group += TAB + "to { coords=%s; label=\"%s\"; nPoints=%s; }\n" % (list(kpts.kpath["kpoints"][path[ki]]), path[ki].replace("\\Gamma", "\\xG"), npoints)
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


def plot_band_structure(output_filename="bands.pdf", spins=[0,1], ylim=[-5, 5], figsize=(12, 8), logfile="sphinx.log"):

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
        os.system("~/software/sphinx/bin/sxpawatomvolume > %s" % filename)
    lines = open(filename).readlines()
    hirshfeld_line = [line for line in lines if "Hirshfeld" in line][0]
    hirshfeld_charges = [float(c) for c in hirshfeld_line.split("[")[1].split("]")[0].split(",")]
    return hirshfeld_charges


def create_charged_structure_file(filename="CONTCAR", convert_from_sphinx=False):
    if convert_from_sphinx or not os.path.isfile(filename):
        if os.path.isfile("relaxedStr.sx"):
            os.system("/u/mashton/software/sphinx/bin/sx2poscar -i relaxedStr.sx -o %s" % filename)
        else: 
            raise("ERROR: No relaxedStr.sx found in %s" % os.getcwd())

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
