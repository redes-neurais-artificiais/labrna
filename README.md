# labrna
Prática de implementação de redes neurais artificiais.

# Instalação
Seguem abaixo algumas alternativas sugeridas de uso.

## Imagem VirtualBox
Contém Debian linux XFCE com vscode. Usuário **u**. Senha **u**.

## Instalação Manual
Em linux (recomendado).
Windows WSL, CygWin e similares.

Instale o poetry, git e dependências do python.
```shell
curl -sSL https://install.python-poetry.org | python3 -
sudo apt install git python3-dev python3-tk
```

Instale o projeto da disciplina.
```shell
    git clone https://github.com/redes-neurais-artificiais/labrna.git
    cd labrna
    poetry install
    poetry env info --path
```

- Instale o vscode conforme seu sistema operacional favorito.
- Configure o vscode:
    - Instale extensões
        - Aperte Control+Shift+X
        - Python Poetry
        - Python (Pylance, Python Environments)
        - Jupyter
        - CoPilot e VS COde Speech (opcional)
    - Aperte Control+Shift+P
    - Escolha "Python: Select Interpreter"
    - Copie o texto produzido pelo comando abaixo
    - ```shell
         poetry env info --path
      ```
    - Cole uma cópia no item "Enter interpreter path"
    - Deve ser algo parecido com "/home/usuario/.cache/pypoetry/virtualenvs/labrna-G-OeFOGY-py3.13"

## Teste
```shell
poetry run python examples/teste.py
```
