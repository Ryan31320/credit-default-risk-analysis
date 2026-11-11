"""Vérifier le notebook sans exécuter les entraînements coûteux."""
import ast
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'notebooks' / 'prediction_credit_default_actualise2.ipynb'


def main():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    count = 0
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != 'code':
            continue
        ast.parse(cell.source, filename=f'{NOTEBOOK.name}:cell-{index}')
        errors = [output for output in cell.get('outputs', []) if output.output_type == 'error']
        if errors:
            raise ValueError(f'Cellule {index} : sortie d’erreur enregistrée')
        count += 1
    print(f'Notebook valide : {len(notebook.cells)} cellules, {count} cellules de code analysées.')


if __name__ == '__main__':
    main()
