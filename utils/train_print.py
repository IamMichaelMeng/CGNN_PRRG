import os


def clear_content(content):
    print(content)
    os.system('clear')


def start_training(model_name):
    print('========================================')
    print(f'{model_name} training started!')


def end_training(model_name):
    print(f'{model_name} training finished!')
    print('========================================\n')
