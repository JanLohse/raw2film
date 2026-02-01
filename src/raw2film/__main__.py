from spectral_film_lut.splash_screen import launch_splash_screen


def run():
    app, splash_screen = launch_splash_screen("Raw2Film")

    from raw2film.gui import MainWindow
    from spectral_film_lut.film_loader import load_ui

    load_ui(MainWindow, splash_screen, app, 0.16)


if __name__ == "__main__":
    run()
