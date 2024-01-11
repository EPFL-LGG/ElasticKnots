import subprocess
import os
import platform

def is_ninja_installed():
    try:
        subprocess.run(["ninja", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def install_ninja():
    print("Ninja not found. Installing Ninja...")
    system_platform = platform.system().lower()
    if system_platform == "linux":
        subprocess.run(["sudo", "apt-get", "install", "ninja-build"])
    elif system_platform == "darwin":
        subprocess.run(["brew", "install", "ninja"])
    elif system_platform == "windows":
        # You may need to customize this for Windows installations
        print("Please install Ninja manually on Windows.")
    else:
        print(f"Ninja installation not supported on {system_platform}. Please install Ninja manually.")

# Check if Ninja is installed
if not is_ninja_installed():
    install_ninja()

# Create the build directory if it doesn't exist and change to it
build_dir = "build"
if not os.path.exists(build_dir):
    os.makedirs(build_dir)
os.chdir(build_dir)

# Run cmake with Ninja generator
cmake_command = ["cmake", "..", "-GNinja"]
subprocess.run(cmake_command, check=True)

# Run ninja to build the C++ library
ninja_command = ["ninja"]
try:
    subprocess.run(ninja_command, check=True)
except subprocess.CalledProcessError:
    # If building fails, it might be because of some overlapping definitions of functions
    # from libigl's predicates and from libtriangle (included in the current version of MeshFEM).
    # We then add a triangle:: scope to the symbols in libtriangle and try building again.
    print("Ninja build failed. Trying redefining symbols in libtriangle...")
    lib_triangle_path = "3rdparty/ElasticRods/3rdparty/MeshFEM/triangle/libtriangle.a"
    objcopy_commands = [
        ["objcopy", "--redefine-sym", "exactinit=triangle::exactinit", lib_triangle_path],
        ["objcopy", "--redefine-sym", "orient3d=triangle::orient3d", lib_triangle_path],
        ["objcopy", "--redefine-sym", "incircle=triangle::incircle", lib_triangle_path]
    ]
    for cmd in objcopy_commands:
        subprocess.run(cmd, check=True)

    # Build again
    print("Trying ninja build again...")
    subprocess.run(ninja_command, check=True)