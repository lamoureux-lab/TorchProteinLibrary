# from PythonInterface import CppLayers, Visualization, Utils, Angles2CoordsDihedral, \
# Angles2BasisDihedral, Coords2Pairs, Coords2RMSD, Pairs2Distributions, Angles2CoordsAB
try:
    from scripts import getAngles
except:
    print 'No scripts'
try:
    from PythonInterface import PDB2Volume, PDB2VolumeLocal, SelectVolume
except:
    print "No PDB2Volume"

try:
    from PythonInterface import cppPDB2Coords, PDB2Coords
except:
    print "No PDB2Coords"

try:
    from PythonInterface import Angles2Coords, Angles2Coords_save
    from PythonInterface import Coords2RMSD
    from PythonInterface import cppPDB2Coords
    from PythonInterface import Coords2CenteredCoords
    from PythonInterface import Coords2TypedCoords
    from PythonInterface import TypedCoords2Volume
except:
    print "No full atom model"

try:
    from PythonInterface import Utils
except:
    print "No utils"

try:
    from PythonInterface import Angles2CoordsDihedral, Coords2RMSDCuda, Angles2Backbone
except:
    print "No C-alpha model"