from time import strftime, localtime
import json
import os

def write_on_log(text, output_path):
    if not is_main_process():
        return

    time = strftime("%Y-%m-%d %H:%M:%S - ", localtime())
    mode = "w" if not os.path.exists(os.path.join(output_path, "log.txt")) else "a"
    with open(os.path.join(output_path, "log.txt"), mode) as file:
        file.write(time + text + "\n")

def write_on_csv(output_path, epoch, iteration, loss, lr, wd, ema="-"):
    if not is_main_process():
        return

    file_path = os.path.join(output_path, "log.csv")
    file_exists = os.path.isfile(file_path)

    with open(file_path, "a") as csv_file:
        if not file_exists:
            csv_file.write("timestamp,epoch,iteration,loss,lr,wd,ema\n")

        csv_file.write(f"{strftime('%Y-%m-%d %H:%M:%S', localtime())},{epoch},{iteration},{loss},{lr},{wd},{ema}\n")

def is_distributed():
    return int(os.environ.get("WORLD_SIZE", "1")) > 1

def is_main_process():
    return int(os.environ.get("RANK", "0")) == 0

def plot_fig(x, x_name, y, y_name, fig_name, output_path):
    if not is_main_process():
        return

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(x, y)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(fig_name)

    figures_path = os.path.join(output_path, "figs")
    os.makedirs(figures_path, exist_ok=True)

    plt.savefig(os.path.join(figures_path, f"{fig_name}.png"))
    plt.close()

def save_json(data, output_path, file_name):
    if not is_main_process():
        return

    with open(os.path.join(output_path, f"{file_name}.json"), "w") as f:
        json.dump(data, f, indent=4)
