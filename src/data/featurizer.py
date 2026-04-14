import torch
from torch_geometric.data import Data
from rdkit import Chem


def get_node_features(atom):
    """
    Extract node features from an RDKit atom.
    Features: atomic number, degree, hybridization, aromaticity, implicit valence, formal charge, radical electrons.
    """
    atomic_num = atom.GetAtomicNum()
    degree = atom.GetDegree()

    try:
        hybridization = int(atom.GetHybridization())
    except:
        hybridization = 0

    is_aromatic = int(atom.GetIsAromatic())
    implicit_valence = atom.GetImplicitValence()
    formal_charge = atom.GetFormalCharge()
    radical_electrons = atom.GetNumRadicalElectrons()

    return [
        float(atomic_num),
        float(degree),
        float(hybridization),
        float(is_aromatic),
        float(implicit_valence),
        float(formal_charge),
        float(radical_electrons),
    ]


def get_edge_features(bond):
    """
    Extract edge features from an RDKit bond.
    Features: bond type (as float), is in ring, is conjugated.
    """
    bond_type = bond.GetBondTypeAsDouble()
    is_in_ring = int(bond.IsInRing())
    is_conjugated = int(bond.GetIsConjugated())

    return [float(bond_type), float(is_in_ring), float(is_conjugated)]


class MolFeaturizer:
    """
    Callable class to convert an RDKit molecule or a PyG Data object with SMILES
    into a PyG Data object with custom featurization.
    """

    def __call__(self, data_or_mol):
        # Handle PyG Data object with SMILES
        if isinstance(data_or_mol, Data) and hasattr(data_or_mol, "smiles"):
            smiles = data_or_mol.smiles
            if isinstance(smiles, list):
                smiles = smiles[0]
            mol = Chem.MolFromSmiles(smiles)
            y = data_or_mol.y
        elif isinstance(data_or_mol, Chem.Mol):
            mol = data_or_mol
            y = None
        elif isinstance(data_or_mol, str):
            mol = Chem.MolFromSmiles(data_or_mol)
            y = None
        else:
            raise ValueError(
                "Input must be an RDKit Mol, a SMILES string, or a PyG Data object with a 'smiles' attribute."
            )

        if mol is None:
            return data_or_mol

        num_nodes = mol.GetNumAtoms()

        node_features = []
        z = []
        for atom in mol.GetAtoms():
            node_features.append(get_node_features(atom))
            z.append(atom.GetAtomicNum())

        edge_indices = []
        edge_features = []

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_features_ij = get_edge_features(bond)

            # Bidirectional edges for undirected graph
            edge_indices.append([i, j])
            edge_indices.append([j, i])
            edge_features.append(edge_features_ij)
            edge_features.append(edge_features_ij)

        x = torch.tensor(node_features, dtype=torch.float)
        z_tensor = torch.tensor(z, dtype=torch.long)

        if len(edge_indices) > 0:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 3), dtype=torch.float)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, z=z_tensor)

        if y is not None:
            data.y = y

        # Preserve SMILES if available
        if isinstance(data_or_mol, Data) and hasattr(data_or_mol, "smiles"):
            data.smiles = data_or_mol.smiles
        elif isinstance(data_or_mol, str):
            data.smiles = data_or_mol

        return data
