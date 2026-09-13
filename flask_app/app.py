<<<<<<< HEAD
from . import create_app
=======
from .__init__ import create_app

>>>>>>> c01d4dc79a76ec62462545a3e06644b9f0580ef6

app = create_app()

if __name__ == "__main__":
    app.run(debug=True)