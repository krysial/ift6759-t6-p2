**IFT6759 - Advanced projects in machine learning**
---

IFT6759 is a course about implementing deep learning theory in applied projects.

# Project 2 - Machine Translation

Setup
---

The typical setup involves installing VSCode with a few extensions:

- SSH remote
- Python
- Codestyle lint

An example of the remote `.vscode/launch.json` file can be seen here:
```javascript
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [

        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "envFile": "${workspaceFolder}/.env",
            "args": ["data/admin_cfg.json", "--model=lrcn", "--real", "--batch-size=2"]
        }
    ]
}
```

An example of the remote `.vscode/settings.json` file can be seen here:
```javascript
{
    "python.pythonPath": "~/ift6759-env/bin/python",
    "python.formatting.autopep8Path": "~/ift6759-env/bin/autopep8",
    "python.linting.pycodestyleEnabled": true,
    "python.linting.enabled": true,
    "python.testing.pytestArgs": [
        "tests"
    ],
    "python.testing.unittestEnabled": false,
    "python.testing.nosetestsEnabled": false,
    "python.testing.pytestEnabled": true,
    "python.dataScience.jupyterServerURI": "local"
}
```


One must make also sure that their ssh keys are synced with the Helios login Node.
An example of a ssh configuration can be seen here:

```bash
Host helios
  HostName helios3.calculquebec.ca
  IdentityFile ~/.ssh/umontreal
  User guest133
  ServerAliveInterval 120
```

Make sure you have setup your local `venv` properly, we recommend setting it up on your local disk on helios doing the following.

1.  `ssh helios`
1.  `module load python/3`
1.  `python -m venv ~/ift6759-env`
1.  `source ~/ift6759-env/bin/activate`
1.  `pip install -r requirements.txt`

This setup allows you to develop quickly and run/train the model on the CPU. However, given that the compute nodes don't have access to the internet. You must setup a python venv that is shared between the login node and the compute node. We opted to do that in the following folder `/project/cq-training-1/project1/teams/team06/ift6759-env/bin/activate`. Hence, if your intent is to use the compute node, you must replace step 4 above with the following line of code.

1. `source /project/cq-training-1/project1/teams/team06/ift6759-env/bin/activate`


Training the model
---

For this project, we have trained our model on Google Colab Pro.

Before training the model there are a few things that need to be completed in order:

1. Upload the whole project on Google Drive

1. Open the notebook `train_language_model.ipynb` from GDrive using Colab then run the whole notebook

1. Open the notebook `train_transformer.ipynb` from GDrive using Colab then run the whole notebook


Evaluating the model
---

In order to evaluate the model, one must make sure that the proper checkpoint files are present in the checkpoints folder.
To get an example of such files, you might want to train the model first see: Training the model section above.

1. Open the notebook `evaluator.ipynb` from GDrive using Colab then run the whole notebook.