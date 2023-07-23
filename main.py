import apis


def main():
    function_to_call = input("Enter function name:")
    func = getattr(apis, function_to_call)
    func()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
