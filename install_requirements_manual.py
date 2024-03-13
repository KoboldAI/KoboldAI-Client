import subprocess
import sys
import platform
import re

def should_install(condition):
    if 'sys_platform' in condition:
        platform_condition = re.search("sys_platform\s*==\s*'(\w+)'", condition)
        if platform_condition:
            current_platform = platform.system().lower()
            if platform_condition.group(1) == 'linux' and not current_platform.startswith('linux'):
                return False
            elif platform_condition.group(1) == 'win32' and not current_platform == 'windows':
                return False
            elif platform_condition.group(1) == 'darwin' and not current_platform == 'darwin':
                return False
    if 'python_version' in condition:
        python_version_condition = re.search("python_version\s*==\s*'([\d\.]+)'", condition)
        if python_version_condition:
            if python_version_condition.group(1) not in sys.version:
                return False
    return True

def install_package(package, extra_index_url):
    print(f"Attempting to install: {package}...")
    command = [sys.executable, '-m', 'pip', 'install', package]
    if extra_index_url:
        command += ['--extra-index-url', extra_index_url]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"Installation of {package} completed.\n")
    else:
        print(f"Failed to install {package}.\n")
    return result

def parse_installation_results(result, package_name):
    if result.returncode == 0 and 'Successfully installed' in result.stdout:
        return 'successful'
    elif result.returncode == 0:
        return 'already_installed'
    else:
        return 'failed'

def print_table(categories, data):
    max_length = max(len(pkg) for group in data.values() for pkg in group)
    column_width = max(max_length, max(len(cat) for cat in categories)) + 2

    print("\n" + "Package Installation Summary".center(column_width * len(categories) + len(categories) - 1, "-"))
    header = " | ".join(cat.center(column_width) for cat in categories)
    print(header)
    print("-" * len(header))

    for i in range(max(len(data[cat]) for cat in categories)):
        row = []
        for cat in categories:
            if i < len(data[cat]):
                row.append(data[cat][i].center(column_width))
            else:
                row.append(" ".center(column_width))
        print(" | ".join(row))

def install_packages_from_file(file_name):
    successful_packages = []
    already_installed_packages = []
    failed_packages = []
    extra_index_url = ""
    with open(file_name, 'r') as file:
        lines = [line.strip() for line in file if line.strip() and not line.startswith('#')]
        total_packages = len(lines)
        print(f"Starting installation of {total_packages} packages...")
        for index, line in enumerate(lines, start=1):
            if line.startswith('--extra-index-url'):
                extra_index_url = line.split(' ')[1]
                continue
            package_name = line.split('==')[0].split(';')[0]
            print(f"Processing {index} of {total_packages}: {package_name}")
            if ';' in line:
                package, condition = line.split(';')
                if should_install(condition.strip()):
                    result = install_package(package, extra_index_url)
                    status = parse_installation_results(result, package_name)
            else:
                result = install_package(line, extra_index_url)
                status = parse_installation_results(result, package_name)

            if status == 'successful':
                successful_packages.append(package_name)
            elif status == 'already_installed':
                already_installed_packages.append(package_name)
            elif status == 'failed':
                failed_packages.append(package_name)

    categories = ["Successfully Installed", "Already Installed", "Failed to Install"]
    data = {
        categories[0]: sorted(successful_packages),
        categories[1]: sorted(already_installed_packages),
        categories[2]: sorted(failed_packages),
    }
    print_table(categories, data)

install_packages_from_file('requirements.txt')