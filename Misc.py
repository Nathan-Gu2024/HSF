import numpy as np
import matplotlib.pyplot as plt
from Bio import AlignIO, SeqIO
from Bio.Phylo.TreeConstruction import DistanceCalculator
import seaborn as sns
from matplotlib.gridspec import GridSpec
from Bio.Blast import NCBIWWW, NCBIXML
from Bio.Align.Applications import MuscleCommandline
from Bio.PDB import *

#
#Takes a fasta file and finds the ATCG makeup of each organism
#@param file fasta, fasta file 
#@return results Dictionary, Dictionary in the form of {organism ID: "A:#A T:#T C:#C G:#G"}
#
def contentCalc(file):
    results = {}
    record_dict = SeqIO.to_dict(SeqIO.parse(file, "fasta"))
    for id in record_dict:
        key = id.toString()
        results[key] = "A: " + record_dict[key].count("A") + " T: " + record_dict[key].count("T") + " C: " + record_dict[key].count("C") + " G: " + record_dict[key].count("G") 
    return results        

#
#Finds the new locations for specific residues in the new protein
#@param contigs String, "chosen contigs" of the new protein
#@param locations List, list of residue numbers to find in corresponding protein
#@return list list of residue locations in the new protein
#
def find_alignment(contigs, locations):
    new_locs = []
    contigs = contigs.split("/")
    A_region = [i[1:].split("-") for i in contigs if "A" in i]
    added_region = [int(i.split("-")[0]) for i in contigs if "A" not in i]
    
    while locations:
        for i in range(len(A_region)):
            region = A_region[i]
            if int(region[0]) <= locations[0] < int(region[1]):  # Check if the location falls in the current region
                loc = (locations[0] - int(region[0]) +
                       sum(added_region[0:i+1]) +  # Sum of added regions before this one
                       sum([int(x[1]) - int(x[0]) for x in A_region[:i]]) +  # Sum of A region lengths before this one
                       i + 1)  # Add one for each inserted A region
                new_locs.append(loc)
                locations.pop(0)  # Remove the processed location
                break  # Stop processing this location
    return  f"New residue locations: {new_locs}"

#
#Runs BLAST and parses the file
#@param file fasta, sequence to be compared to for BLAST
#@param program String, which program to run (blastp, blastx, etc...); will except an error if invalid
#@param db String, which database to compared to (nt for nucleotide DB, nr for protein DB)
#@param num Integer, top num hits for BLAST
#
def BLAST(file, program, db, num):
    seq = str(SeqIO.read(file, "fasta"))
    try: 
        result = NCBIWWW.qblast(program, db, seq)
    except Exception as e:
        print("Invalid input")
    blast_res = NCBIXML.read(result)

    for alignment in blast_res.alignments[:num]:  
        for hsp in alignment.hsps:
            print(f"  Score: {hsp.score}")
            print(f"  E-value: {hsp.expect}")
            print(f"  Identity: {hsp.identities}/{hsp.align_length} ({100 * hsp.identities / hsp.align_length:.1f}%)")
            print(f"  Query: {hsp.query[0:75]}...")
            print(f"  Match: {hsp.match[0:75]}...")
            print(f"  Subject: {hsp.sbjct[0:75]}...")
            break  

#MSA visualizer
def visualize_msa(filepath):
    muscle_cline = MuscleCommandline(filepath, out="aligned.fasta")
    muscle_cline()
    # Read alignment file
    alignment = AlignIO.read("aligned.fasta", filepath.split('.')[-1])
    
    # Verify all sequences are same length
    if len(set(len(rec.seq) for rec in alignment)) > 1:
        raise ValueError("Input sequences must be pre-aligned with equal lengths")

    
    # Calculate Shannon Entropy
    def shannon_entropy(column):
        counts = {}
        for aa in column:
            counts[aa] = counts.get(aa, 0) + 1
        entropy = 0.0
        total = sum(counts.values())
        for count in counts.values():
            p = count / total
            entropy -= p * np.log2(p) if p > 0 else 0
        return entropy
    
    entropy = [shannon_entropy(alignment[:, i]) for i in range(alignment.get_alignment_length())]
    
    # Calculate Distance Matrix
    dm = DistanceCalculator('identity').get_distance(alignment)
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 1, height_ratios=[1, 0.5, 1])
    
    # Sequence Alignment Map
    ax0 = plt.subplot(gs[0])
    aa_colors = {
        'A': '#FF0000', 'V': '#FF6600', 'L': '#FFCC00', 'I': '#FFFF00',
        'M': '#00FF00', 'F': '#00FF66', 'W': '#00FFCC', 'Y': '#00FFFF',
        'D': '#0000FF', 'H': '#6600FF', 'N': '#CC00FF', 'E': '#FF00FF',
        'K': '#FFFFFF', 'Q': '#FF0066', 'R': '#FF00CC', 'S': '#FF66FF',
        'T': '#FFCCFF', 'C': '#990000', 'G': '#990066', 'P': '#9900CC',
        '-': '#000000'  # Gaps
    }
    
    for i, record in enumerate(alignment):
        for j, aa in enumerate(record.seq):
            ax0.add_patch(plt.Rectangle((j, i), 1, 1, 
                                      color=aa_colors.get(aa.upper(), '#FFFFFF'), 
                                      ec='gray'))
    
    ax0.set_xlim(0, alignment.get_alignment_length())
    ax0.set_ylim(0, len(alignment))
    ax0.set_yticks(np.arange(len(alignment)) + 0.5)
    ax0.set_yticklabels([record.id for record in alignment])
    ax0.set_title('Sequence Alignment Map')
    
    # Conservation Plot
    ax1 = plt.subplot(gs[1])
    ax1.plot(entropy, color='blue')
    ax1.set_ylabel('Shannon Entropy')
    ax1.set_xlabel('Alignment Position')
    ax1.set_title('Conservation Score')
    ax1.grid(alpha=0.3)
    
    # Pairwise Distance Heatmap
    ax2 = plt.subplot(gs[2])
    sns.heatmap(np.array(dm.matrix), 
                annot=True, 
                xticklabels=[record.id[:10] for record in alignment],
                yticklabels=[record.id[:10] for record in alignment],
                cmap='viridis', 
                ax=ax2)
    ax2.set_title('Pairwise Distance Heatmap')
    
    plt.tight_layout()
    plt.show()

# visualize_msa("/Users/nathangu/Desktop/HSF/Scripts + Tools/test2.fasta")


def compare_dist(pdb1, pdb2, targets = []):
        # Parse structures
    parser = PDBParser()
    ref_structure = parser.get_structure("REF", pdb1)
    alt_structure = parser.get_structure("ALT", pdb2)
    
    # Get all CA atoms for global alignment
    ref_atoms = []
    alt_atoms = []
    
    for ref_res, alt_res in zip(Selection.unfold_entities(ref_structure, 'R'), Selection.unfold_entities(alt_structure, 'R')):
        if ('CA' in ref_res and 'CA' in alt_res):
            ref_atoms.append(ref_res['CA'])
            alt_atoms.append(alt_res['CA'])
    
    # Perform global alignment
    sup = Superimposer()
    sup.set_atoms(ref_atoms, alt_atoms)
    sup.apply(alt_structure.get_atoms())
    
    results = {'global_rmsd': sup.rms, 'residue_distances': {}}
    if targets:
        for target in targets:
            try:
                # Find the residues
                ref_res = None
                alt_res = None
                
                for residue in Selection.unfold_entities(ref_structure, 'R'):
                    if residue.id[1] == target:
                        ref_res = residue
                        break
                        
                for residue in Selection.unfold_entities(alt_structure, 'R'):
                    if residue.id[1] == target:
                        alt_res = residue
                        break
                
                if ref_res and alt_res and 'CA' in ref_res and 'CA' in alt_res:
                    distance = ref_res['CA'] - alt_res['CA']
                    results['residue_distances'][target] = distance
                else:
                    results['residue_distances'][target] = None
                    
            except Exception as e:
                print(f"Error processing residue {target}: {str(e)}")
                results['residue_distances'][target] = None

    return results


def vector_orientation(residue, atom1='CA', atom2='CB'):
    if atom1 in residue and atom2 in residue:
        a = residue[atom1].get_vector()
        b = residue[atom2].get_vector()
        return (b - a).normalized()
    else:
        return None

    # Parse two aligned structures
    parser = PDBParser()
    struct1 = parser.get_structure("ref", "ref.pdb")
    struct2 = parser.get_structure("alt", "alt.pdb")

    # Superimpose on CA atoms
    chain1 = struct1[0]['A']
    chain2 = struct2[0]['A']
    atoms1 = [res['CA'] for res in chain1 if 'CA' in res]
    atoms2 = [res['CA'] for res in chain2 if 'CA' in res]

    from Bio.PDB import Superimposer
    sup = Superimposer()
    sup.set_atoms(atoms1, atoms2)
    sup.apply(chain2.get_atoms())  # aligns structure 2 to 1

    # Compare orientations
    print("Comparing residue orientation (CA->CB vectors)...\n")
    for r1, r2 in zip(chain1, chain2):
        if r1.id[0] != ' ' or r2.id[0] != ' ':
            continue  # Skip non-standard residues
        v1 = vector_orientation(r1)
        v2 = vector_orientation(r2)
        if v1 and v2:
            angle_rad = v1.angle(v2)
            angle_deg = np.degrees(angle_rad)
            print(f"Residue {r1.resname} {r1.id[1]}: Orientation difference = {angle_deg:.2f}°")
