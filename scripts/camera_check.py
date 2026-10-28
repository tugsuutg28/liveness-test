from core.device_inspection import inspect_macos
if __name__=='__main__':
    print('Informational camera inventory (NOT trusted provenance):')
    for device in inspect_macos():print(device)
