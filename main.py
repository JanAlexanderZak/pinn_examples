from src.euler_bernoulli_beam.executor import main as euler_bernoulli_beam
from src.heat_eq_1d.executor import main as heat_eq_1d
from src.heat_eq_2d.executor import main as heat_eq_2d
from src.inverse_burgers.executor import main as inverse_burgers
from src.moseley_oscillator.executor import main as moseley_oscillator
from src.navier_stokes_kovasznay.executor import main as navier_stokes_kovasznay
from src.projectile_trajectory.executor import main as projectile_trajectory
from src.raissi_allen_cahn.executor import main as raissi_allen_cahn
from src.raissi_burgers.executor import main as raissi_burgers
from src.thick_walled_cylinder.executor import main as thick_walled_cylinder
from src.wave_eq_1d.executor import main as wave_eq_1d

EXAMPLES = {
    #"euler_bernoulli_beam": (euler_bernoulli_beam, (15000,)),
    "heat_eq_1d": (heat_eq_1d, (10000,)),
    "heat_eq_2d": (heat_eq_2d, (20002,)),
    "inverse_burgers": (inverse_burgers, (20500,)),
    "moseley_oscillator": (moseley_oscillator, ()),
    "navier_stokes_kovasznay": (navier_stokes_kovasznay, (15000,)),
    "projectile_trajectory": (projectile_trajectory, (8000,)),
    "raissi_allen_cahn": (raissi_allen_cahn, (200000,)),
    "raissi_burgers": (raissi_burgers, (20500,)),
    "thick_walled_cylinder": (thick_walled_cylinder, (12000,)),
    "wave_eq_1d": (wave_eq_1d, (20000,)),
}

if __name__ == "__main__":
    for name, (fn, args) in EXAMPLES.items():
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print(f"{'='*60}\n")
        fn(*args)
        print(f"\nFinished: {name}")
