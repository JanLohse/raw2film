on:
  release:
    types: [published]

name: Build Release Artifacts

jobs:
  build:
    strategy:
      matrix:
        os: [windows-latest, ubuntu-latest]

    runs-on: ${{ matrix.os }}

    permissions:
      contents: write

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.13"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install pyinstaller build
          pip install -e .

      - name: Set version
        shell: bash
        run: echo "VERSION=${GITHUB_REF_NAME#v}" >> $GITHUB_ENV

      - name: Build with PyInstaller
        run: |
          pyinstaller raw2film.spec

      # Windows artifact
      - name: Package Windows EXE
        if: matrix.os == 'windows-latest'
        shell: bash
        run: |
          mv dist/Raw2Film.exe Raw2Film-${VERSION}.exe

      # Linux artifact (AppImage-like single binary)
      - name: Package Linux binary
        if: matrix.os == 'ubuntu-latest'
        shell: bash
        run: |
          chmod +x dist/Raw2Film
          mv dist/Raw2Film Raw2Film-${VERSION}.AppImage

      - name: Upload release assets
        uses: softprops/action-gh-release@v2
        with:
          files: |
            Raw2Film-${{ env.VERSION }}.exe
            Raw2Film-${{ env.VERSION }}.AppImage
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
