from helmet_violation_monitoring_gui import run_lpr


def main():
    af = open('api_key.txt', 'r')
    api_key = af.read()
    api_key = api_key.strip()
    af.close()
    run_lpr(api_key)


if __name__ == "__main__":
    main()
