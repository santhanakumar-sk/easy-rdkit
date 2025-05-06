import os
from flask import Flask, request, jsonify, send_file, session
from flask_cors import CORS
from rdkit import Chem
from rdkit.Chem import Descriptors, MolSurf, Crippen, EState
import pandas as pd
import io
import pubchempy as pcp
from collections import OrderedDict

app = Flask(__name__, static_folder='static', static_url_path='')
app.secret_key = 'your-secret-key'  # Required for session; replace with a secure key in production
CORS(app)

# Define descriptor order
DESCRIPTOR_ORDER = [
'fr_C_O_noCOO', 'MaxEStateIndex', 'Chi4v', 'fr_Ar_COO', 'Chi4n', 'SMR_VSA4', 'fr_urea',
'fr_para_hydroxylation', 'fr_barbitur', 'fr_Ar_NH', 'fr_halogen', 'fr_dihydropyridine',
'fr_priamide', 'Chi0n', 'fr_Al_COO', 'fr_guanido', 'MinPartialCharge', 'fr_furan',
'fr_morpholine', 'fr_term_acetylene', 'SlogP_VSA6', 'fr_amidine', 'fr_benzodiazepine',
'ExactMolWt', 'SlogP_VSA1', 'MolWt', 'NumHDonors', 'fr_hdrzine', 'NumAromaticRings',
'fr_quatN', 'NumSaturatedHeterocycles', 'NumAliphaticHeterocycles', 'fr_benzene',
'fr_phos_acid', 'fr_sulfone', 'VSA_EState10', 'fr_aniline', 'fr_N_O', 'fr_sulfonamd',
'fr_thiazole', 'TPSA', 'fr_piperzine', 'SMR_VSA10', 'PEOE_VSA13', 'PEOE_VSA12',
'PEOE_VSA11', 'PEOE_VSA10', 'BalabanJ', 'fr_lactone', 'Chi3v', 'Chi2n', 'EState_VSA10',
'EState_VSA11', 'HeavyAtomMolWt', 'Chi0', 'Chi1', 'NumAliphaticRings', 'MolLogP',
'fr_nitro', 'fr_Al_OH', 'fr_azo', 'NumAliphaticCarbocycles', 'fr_C_O', 'fr_ether',
'fr_phenol_noOrthoHbond', 'RingCount', 'fr_alkyl_halide', 'NumValenceElectrons',
'fr_aryl_methyl', 'MinEStateIndex', 'HallKierAlpha', 'fr_C_S', 'fr_thiocyan', 'fr_NH0',
'VSA_EState4', 'fr_nitroso', 'VSA_EState6', 'VSA_EState7', 'VSA_EState1', 'VSA_EState2',
'VSA_EState3', 'fr_HOCCN', 'BertzCT', 'SlogP_VSA12', 'VSA_EState9', 'SlogP_VSA10',
'SlogP_VSA11', 'fr_COO', 'NHOHCount', 'fr_unbrch_alkane', 'NumSaturatedRings',
'MaxPartialCharge', 'fr_methoxy', 'fr_amide', 'SlogP_VSA8', 'SlogP_VSA9', 'SlogP_VSA4',
'SlogP_VSA5', 'NumAromaticCarbocycles', 'SlogP_VSA7', 'fr_Imine', 'SlogP_VSA2',
'SlogP_VSA3', 'fr_phos_ester', 'fr_NH2', 'MinAbsPartialCharge', 'SMR_VSA3',
'NumHeteroatoms', 'fr_NH1', 'fr_ketone_Topliss', 'fr_SH', 'LabuteASA', 'fr_thiophene',
'Chi3n', 'fr_imidazole', 'fr_nitrile', 'SMR_VSA2', 'SMR_VSA1', 'SMR_VSA7', 'SMR_VSA6',
'EState_VSA8', 'EState_VSA9', 'EState_VSA6', 'fr_nitro_arom', 'SMR_VSA9', 'EState_VSA5',
'EState_VSA2', 'fr_Ndealkylation2', 'fr_Ndealkylation1', 'EState_VSA1', 'PEOE_VSA14',
'Kappa3', 'Ipc', 'fr_diazo', 'Kappa2', 'fr_Ar_N', 'fr_Nhpyrrole', 'EState_VSA7',
'MolMR', 'VSA_EState5', 'EState_VSA4', 'fr_COO2', 'fr_prisulfonamd', 'fr_oxime',
'SMR_VSA8', 'fr_isocyan', 'EState_VSA3', 'Chi2v', 'HeavyAtomCount', 'fr_aldehyde',
'SMR_VSA5', 'NumHAcceptors', 'fr_lactam', 'fr_allylic_oxid', 'VSA_EState8', 'fr_oxazole',
'fr_piperdine', 'fr_Ar_OH', 'NumRadicalElectrons', 'fr_sulfide', 'fr_alkyl_carbamate',
'NOCount', 'Chi1n', 'MaxAbsEStateIndex', 'PEOE_VSA7', 'PEOE_VSA6', 'PEOE_VSA5',
'PEOE_VSA4', 'PEOE_VSA3', 'PEOE_VSA2', 'PEOE_VSA1', 'NumSaturatedCarbocycles',
'fr_imide', 'FractionCSP3', 'Chi1v', 'fr_Al_OH_noTert', 'fr_epoxide', 'fr_hdrzone',
'fr_isothiocyan', 'NumAromaticHeterocycles', 'fr_bicyclic', 'Kappa1', 'MinAbsEStateIndex',
'fr_phenol', 'fr_ester', 'PEOE_VSA9', 'fr_azide', 'PEOE_VSA8', 'fr_pyridine',
'fr_tetrazole', 'fr_ketone', 'fr_nitro_arom_nonortho', 'Chi0v', 'fr_ArN',
'NumRotatableBonds', 'MaxAbsPartialCharge'
]

def get_iupac_and_compound_name(smiles, custom_compound_name=None):
    try:
        compounds = pcp.get_compounds(smiles, namespace='smiles')
        if not compounds:
            return 'Not found in PubChem', custom_compound_name or 'Unknown'

        compound = compounds[0]
        iupac_name = compound.iupac_name if compound.iupac_name else 'Not available'
        
        if custom_compound_name:
            compound_name = custom_compound_name
        else:
            compound_name = compound.synonyms[0] if compound.synonyms else smiles
        
        return iupac_name, compound_name
    except Exception as e:
        return 'PubChem API error', custom_compound_name or 'Unknown'

def calculate_descriptors(mol, smiles, custom_compound_name=None):
    descriptors = OrderedDict()

    iupac_name, compound_name = get_iupac_and_compound_name(smiles, custom_compound_name)
    descriptors['Compound_Name'] = compound_name
    descriptors['IUPAC_Name'] = iupac_name

    descriptors['SMILES'] = smiles

    for desc in DESCRIPTOR_ORDER:
        try:
            descriptors[desc] = getattr(Descriptors, desc)(mol) if desc in Descriptors.__dict__ else \
                               getattr(MolSurf, desc)(mol) if desc in MolSurf.__dict__ else \
                               getattr(Crippen, desc)(mol) if desc in Crippen.__dict__ else \
                               getattr(EState, desc)(mol) if desc in EState.__dict__ else 0
        except Exception:
            descriptors[desc] = 0

    return descriptors

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/process_single', methods=['POST'])
def process_single_smiles():
    session.pop('csv_buffer', None)  # Clear previous session data
    data = request.get_json()
    smiles = data.get('smiles', '').strip()

    if not smiles:
        return jsonify({'error': 'No SMILES provided'}), 400

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return jsonify({'error': f'Invalid SMILES: {smiles}'}), 400
        
        descriptors = calculate_descriptors(mol, smiles)
        descriptors_list = [descriptors]

        df = pd.DataFrame(descriptors_list)
        cols = ['SMILES', 'Compound_Name', 'IUPAC_Name'] + DESCRIPTOR_ORDER
        df = df[cols].fillna(0)

        ordered_descriptors = OrderedDict([('SMILES', descriptors['SMILES']), ('Compound_Name', descriptors['Compound_Name']), ('IUPAC_Name', descriptors['IUPAC_Name'])] + [(key, descriptors[key]) for key in DESCRIPTOR_ORDER])
        ordered_descriptors_list = [ordered_descriptors]

        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)

        session['csv_buffer'] = output.getvalue()

        return jsonify({'descriptors': ordered_descriptors_list})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/process', methods=['POST'])
def process_file():
    session.pop('csv_buffer', None)  # Clear previous session data
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if not file.filename:
        return jsonify({'error': 'No file selected'}), 400

    filename = file.filename.lower()
    descriptors_list = []

    try:
        if filename.endswith('.txt'):
            smiles = file.read().decode('utf-8').strip()
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return jsonify({'error': f'Invalid SMILES: {smiles}'}), 400
            descriptors = calculate_descriptors(mol, smiles)
            descriptors_list.append(descriptors)
        elif filename.endswith('.csv'):
            df = pd.read_csv(file, header=None)
            for idx, smiles in enumerate(df[0]):
                smiles = str(smiles).strip()
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    print(f"Warning: Invalid SMILES at row {idx+1}: {smiles}")
                    continue
                descriptors = calculate_descriptors(mol, smiles)
                descriptors_list.append(descriptors)
        else:
            return jsonify({'error': 'Unsupported file type. Use .txt or .csv'}), 400

        if not descriptors_list:
            return jsonify({'error': 'No valid SMILES strings found'}), 400

        df = pd.DataFrame(descriptors_list)
        cols = ['SMILES', 'Compound_Name', 'IUPAC_Name'] + DESCRIPTOR_ORDER
        df = df[cols].fillna(0)

        ordered_descriptors_list = []
        for desc in descriptors_list:
            ordered_desc = OrderedDict([('SMILES', desc['SMILES']), ('Compound_Name', desc['Compound_Name']), ('IUPAC_Name', desc['IUPAC_Name'])] + [(key, desc[key]) for key in DESCRIPTOR_ORDER])
            ordered_descriptors_list.append(ordered_desc)

        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)

        session['csv_buffer'] = output.getvalue()

        return jsonify({'descriptors': ordered_descriptors_list})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download')
def download_csv():
    csv_buffer = session.get('csv_buffer')
    if not csv_buffer:
        return jsonify({'error': 'No CSV data available'}), 400
    return send_file(
        io.BytesIO(csv_buffer.encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name='descriptors.csv'
    )

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))