python setup.py bdist_wheel --universal
twine upload dist/*
rm -r dist
rm -r build
rm -r e2eqavn.egg-info