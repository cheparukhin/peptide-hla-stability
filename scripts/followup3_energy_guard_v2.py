"""Use the reviewed deadline guard for the separately named recovery app."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import followup3_energy_guard as guard

guard.APP_NAME = 'pepstab-followup3-repaired-energy-v2'

if __name__ == '__main__':
    guard.main()
