from setuptools import setup

setup(name='Aide-memoire',
      version='0.3.0',
      description='Recommendation system to suggest possible links between a pullrequest and open issues',
      url='N/A',
      author='Profir-Petru Partachi',
      author_email='profir.p.partachi@gmail.com',
      license='MIT',
      packages=['gitMine', 'Util', 'Prediction', 'backend'],
      install_requires=['jsonpickle', 'scikit-learn', 'gensim', 'numpy', 'pygithub', 'pytz', 'dateutil'],
      zip_safe=False)
