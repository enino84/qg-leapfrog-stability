# Publicar qg-python en PyPI

Esta carpeta es tu repo `qg-leapfrog-stability` con lo mínimo para que
`pip install qg-python` funcione. Nada del flujo del paper cambia:
`run_all.sh`, `analysis/`, `experiments/` siguen igual.

## Qué se agregó o cambió

- `pyproject.toml`   metadatos del paquete (nombre `qg-python`, versión 1.0.0,
                     dependencias numpy y scipy, extras `plots`, `netcdf`, `all`,
                     comando `qg-python`).
- `MANIFEST.in`      excluye paper/, data/, figures/ del paquete fuente.
- `qg_model.py`      `__version__ = "1.0.0"`; el bloque de línea de comandos quedó
                     dentro de `main()` para exponer el comando `qg-python`;
                     la matriz de Helmholtz se pasa en formato CSC (quita un warning).
- `README.md`        sección "Install".
- `CITATION.cff` y README: el DOI `10.1002/gdj3.XXXX` ahora es `10.1002/gdj3.70107`.
- `dist/`            la rueda y el tar.gz ya construidos y verificados con twine.

Ya está probado: se instaló en un entorno limpio, `from qg_model import QGModel, QGParams`
corre una simulación y el comando `qg-python --help` responde.

## Pasos

1. Copia estos archivos sobre tu repo y haz commit:
       git add -A && git commit -m "Package as qg-python 1.0.0" && git push

2. Token de PyPI: en https://pypi.org/manage/account/token/ crea uno
   (scope "Entire account" la primera vez; después puedes limitarlo al proyecto).

3. Sube lo que ya está en dist/:
       pip install twine
       twine upload dist/*
   Usuario: __token__      Contraseña: el token (empieza por pypi-)

   Si prefieres probar antes en el índice de pruebas:
       twine upload --repository testpypi dist/*
       pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple qg-python

4. Comprueba en cualquier máquina:
       pip install qg-python
       python -c "from qg_model import QGModel, QGParams; print('ok')"
       qg-python --help

## Para versiones siguientes

Cambia `version` en pyproject.toml y `__version__` en qg_model.py, y luego:
       rm -rf dist && python -m build && twine upload dist/*
