import datetime
import os
import shutil
import sqlite3
import time
from tkinter import Tk, Label, Button, mainloop, Toplevel, Entry, filedialog, ttk, messagebox

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas as canvas1

from deep_license_plate_recognition.plate_recognition import recognition_api
from keras_yolo3.yolo import detect_video, YOLO, CROPPED_IMAGES_DIRECTORY

VEHICLES_DB_FILE = 'vehicles.db'
VIOLATIONS_IMAGES_DIR = os.path.join('keras_yolo3', 'violations')
CHALLANS_DIR = "challans"

# create folders if not present
for dirname in [CROPPED_IMAGES_DIRECTORY, VIOLATIONS_IMAGES_DIR, CHALLANS_DIR]:
    if not os.path.exists(dirname):
        os.makedirs(dirname)

master = Tk()


def open_database_entry_window():
    new_window = Toplevel(master)
    new_window.title("Database Entry")

    center_window(new_window, width=500, height=400)

    label_font = ("Arial", 12)
    entry_font = ("Arial", 12)
    button_font = ("Arial", 12, "bold")

    new_window.grid_columnconfigure(0, weight=1)
    new_window.grid_columnconfigure(1, weight=1)

    l0 = Label(new_window, text="Database Entry", font=("Arial", 16, "bold"))
    l0.grid(row=0, column=0, columnspan=2, pady=20)

    l1 = Label(new_window, text="License Plate", font=label_font)
    l1.grid(row=1, column=0, padx=20, pady=10, sticky="e")
    e1 = Entry(new_window, font=entry_font, width=30)
    e1.grid(row=1, column=1, padx=20, pady=10, sticky="w")

    l2 = Label(new_window, text="Name on Registration", font=label_font)
    l2.grid(row=2, column=0, padx=20, pady=10, sticky="e")
    e2 = Entry(new_window, font=entry_font, width=30)
    e2.grid(row=2, column=1, padx=20, pady=10, sticky="w")

    l3 = Label(new_window, text="Address", font=label_font)
    l3.grid(row=3, column=0, padx=20, pady=10, sticky="e")
    e3 = Entry(new_window, font=entry_font, width=30)
    e3.grid(row=3, column=1, padx=20, pady=10, sticky="w")

    btn4 = Button(new_window, text="Submit Record", font=button_font,
                  command=lambda: enter_records(e1, e2, e3))
    btn4.grid(row=5, column=0, columnspan=2, pady=20, padx=20)


def enter_records(e1, e2, e3):
    conn = sqlite3.connect(VEHICLES_DB_FILE)
    cursor = conn.cursor()
    cursor.execute("insert into vehicles values('{}','{}','{}')".format(e1.get(), e2.get(), e3.get()))
    conn.commit()
    print("Records added...")
    conn.close()


def run_program():
    root = Tk()
    root.withdraw()

    input_video = filedialog.askopenfilename(
        title="Select Input Video File",
        filetypes=(("MP4 files", "*.mp4"), ("All files", "*.*"))
    )

    output_folder = filedialog.askdirectory(
        title="Select Output Folder"
    )

    if input_video and output_folder:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_video = os.path.join(output_folder, f"demo_output_{timestamp}.mp4")
        detect_video(YOLO(score=0.7), input_video, output_video)
    else:
        labelr = Label(master, text="Error: No file or folder selected!", font=("Arial", 14, "bold"), fg="red")
        labelr.pack(pady=10)
        print("No file or folder selected!")


def generate_challans():
    # gj1md1577
    # gj01uk5541
    af = open('api_key.txt', 'r')
    api_key = af.read()
    api_key = api_key.strip()
    af.close()

    run_lpr(api_key)  # generates and saves images of violations with their license plate in VIOLATIONS_IMAGES_DIR
    for violation in os.listdir(VIOLATIONS_IMAGES_DIR):
        license_plate_str = violation.split('.')[0]
        conn = sqlite3.connect(VEHICLES_DB_FILE)
        cursor = conn.cursor()
        cursor.execute(f"select * from vehicles where plate='{license_plate_str}';")
        result = cursor.fetchone()
        if result is None:
            print(f"No database entry found for license plate {license_plate_str}, skipping...")
            continue
        print(f"Database entry found for license plate {license_plate_str}, generating challan...")
        name = result[1]
        address = result[2]

        conn.close()

        canvas = canvas1.Canvas(os.path.join(CHALLANS_DIR, license_plate_str + ".pdf"), pagesize=letter)
        canvas.setLineWidth(.3)
        canvas.setFont('Helvetica', 12)

        canvas.drawString(30, 750, 'CHALLAN')
        canvas.drawString(30, 730, 'TRAFFIC POLICE DEPARTMENT, BENGALURU')
        canvas.drawString(30, 710, str(datetime.datetime.now()))
        canvas.line(10, 707, 580, 707)

        canvas.drawString(30, 650, 'VIOLATION:')
        canvas.drawString(300, 650, "NOT WEARING HELMET")

        canvas.drawString(30, 600, 'LICENSE PLATE:')
        canvas.drawString(300, 600, license_plate_str)

        canvas.drawString(30, 550, 'NAME:')
        canvas.drawString(300, 550, name)

        canvas.drawString(30, 500, 'ADDRESS:')
        canvas.drawString(300, 500, address)

        canvas.drawString(30, 450, 'FINE:')
        canvas.drawString(300, 450, 'Rs. 500')

        canvas.drawImage(os.path.join(VIOLATIONS_IMAGES_DIR, violation), 30, 150, 250, 250, preserveAspectRatio=True,
                         mask='auto')

        canvas.save()

    labelr = Label(master, text="All challans generated, stored in folder " +
                                str(os.path.abspath(CHALLANS_DIR)))

    labelr.pack(pady=10)
    print("Challans generated.")


def run_lpr(api_key: str):
    for cropped_image in os.listdir(CROPPED_IMAGES_DIRECTORY):
        if "unknown" in cropped_image.lower():
            print(f"Skipping {cropped_image}, already processed as unknown license plate")
            continue
        cropped_image_path = os.path.join(CROPPED_IMAGES_DIRECTORY, cropped_image)
        with open(cropped_image_path, 'rb') as fp:
            api_res = recognition_api(fp, api_key=api_key)

        if 'results' not in api_res:
            print("Full API response = ", api_res)
            raise ValueError("LPR API error encountered")

        filename, file_extension = os.path.splitext(cropped_image)
        if len(api_res['results']) == 0:
            shutil.move(cropped_image_path,
                        os.path.join(CROPPED_IMAGES_DIRECTORY, filename + "_unknown" + file_extension))
            continue

        license_plate_str = api_res['results'][0]['plate']
        shutil.move(cropped_image_path,
                    os.path.join(VIOLATIONS_IMAGES_DIR, license_plate_str.strip() + file_extension))


def center_window(window, width=600, height=400):
    window.update_idletasks()
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()

    x = (screen_width // 2) - (width // 2)
    y = (screen_height // 2) - (height // 2)

    window.geometry(f'{width}x{height}+{x}+{y}')


def view_database():
    db_window = Toplevel()
    db_window.title("Vehicle Database")

    center_window(db_window, 500, 300)

    conn = sqlite3.connect(VEHICLES_DB_FILE)
    cursor = conn.cursor()

    tree = ttk.Treeview(db_window, columns=("License Plate", "Name", "Address"), show="headings")
    tree.heading("License Plate", text="License Plate")
    tree.heading("Name", text="Name")
    tree.heading("Address", text="Address")
    tree.column("License Plate", anchor='center', width=100)
    tree.column("Name", anchor='center', width=150)
    tree.column("Address", anchor='center', width=200)

    scrollbar = ttk.Scrollbar(db_window, orient="vertical", command=tree.yview)
    tree.configure(yscroll=scrollbar.set)  # type: ignore
    scrollbar.pack(side="right", fill="y")
    tree.pack(fill="both", expand=True)

    try:
        cursor.execute("SELECT plate, name, address FROM vehicles")
        rows = cursor.fetchall()
        for row in rows:
            tree.insert('', 'end', values=row)
    except sqlite3.Error as e:
        messagebox.showerror("Database Error", str(e))
    finally:
        conn.close()


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    center_window(master, 600, 400)
    label = Label(master,
                  text="Helmet Violation Monitoring GUI", font=("Arial", 16, "bold"))
    label.pack(pady=10)
    # a button widget which will open a
    # new window on button click
    btn = Button(master,
                 text="Click to open database entry",
                 command=open_database_entry_window)
    btn.pack(pady=10)
    btn2 = Button(master,
                  text="Click to run helmet detection program",
                  command=run_program)
    btn2.pack(pady=10)

    btn3 = Button(master,
                  text="Click to generate challans",
                  command=generate_challans)
    btn3.pack(pady=10)

    btn4 = Button(master, text="View Database", command=view_database)
    btn4.pack(pady=10)

    mainloop()


if __name__ == "__main__":
    main()
