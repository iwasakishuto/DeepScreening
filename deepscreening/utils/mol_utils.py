#coding: utf-8
from rdkit import Chem
from kerasy.datasets import zinc
import pubchempy as pcp

# ================
# Type conversion
# ================

def name2SMILES(name):
    return pcp.get_compounds(identifier=name, namespace='name')[0].isomeric_smiles

def zincid2SMILES(id):
    return zinc.getSMILES(id)

def SMILES2QED(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.QED.qed(mol)

def SMILES2mol(smiles):
    return Chem.MolFromSmiles(smiles)

def mol2SMILES(mol):
    return Chem.MolToSmiles(mol_obj)

def canonicalizeSMILES(smiles):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True, canonical=True)

# ================
#
# ================
