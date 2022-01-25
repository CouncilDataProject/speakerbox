# Contributing

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

## Developer Installation

If something goes wrong at any point during installing the library please see how
[our CI/CD on GitHub Actions](.github/workflows/build-main.yml) installs and builds the
project as it will always be the most up-to-date.

## Get Started!

Ready to contribute? Here's how to set up `speakerbox` for local development.

1. Fork the `speakerbox` repo on GitHub.

2. Clone your fork locally:

    ```bash
    git clone git@github.com:{your_name_here}/speakerbox.git
    ```

3. Install the project in editable mode. (It is also recommended to work in a virtualenv or anaconda environment):

    ```bash
    cd speakerbox/
    pip install -e .[dev]
    ```

4. Create a branch for local development:

    ```bash
    git checkout -b {your_development_type}/short-description
    ```

    Ex: feature/read-tiff-files or bugfix/handle-file-not-found<br>
    Now you can make your changes locally.

5. When you're done making changes, check that your changes pass linting and
   tests, including testing other Python versions with make:

    ```bash
    make build
    ```

6. Commit your changes and push your branch to GitHub:

    ```bash
    git add .
    git commit -m "Your detailed description of your changes."
    git push origin {your_development_type}/short-description
    ```

7. Submit a pull request through the GitHub website.

## Deploying

A reminder for the maintainers on how to deploy.
Make sure all your changes are committed.
Then run:

```bash
$ bump2version patch # possible: major / minor / patch
$ git push
$ git push --tags
```

This will release a new package version on Git + GitHub and publish to PyPI.
