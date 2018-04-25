from setuptools import setup

setup(name='Aide-memoire',
      version='0.5.0',
      description='Recommendation system to suggest possible links between a pullrequest and open issues',
      url='N/A',
      author='Profir-Petru Partachi',
      author_email='profir.p.partachi@gmail.com',
      license='MIT',
      packages=['gitMine', 'Util', 'Prediction', 'backend'],
      entry_points={
          'console_scripts': [
              'am-backend-run = backend.backend:main',
              'am-model-generate = backend.generate_model:main'
          ],
      },
      install_requires=['jsonpickle', 'scikit-learn', 'gensim', 'numpy', 'pygithub', 'pytz', 'pandas', 'scikit-garden'],
      zip_safe=False)
