import datetime
import os
import sqlite3
from tkinter import Tk, Label, Button, mainloop, Toplevel, Entry

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas as canvas1

from keras_yolo3.yolo import detect_video, YOLO, CROPPED_IMAGES_DIRECTORY
from run_lpr import run_lpr

VEHICLES_DB_FILE = 'vehicles.db'
VIOLATIONS_IMAGES_DIR = os.path.join('keras_yolo3', 'violations')
CHALLANS_DIR = "challans"

# create folders if not present
for dirname in [CROPPED_IMAGES_DIRECTORY, VIOLATIONS_IMAGES_DIR, CHALLANS_DIR]:
    if not os.path.exists(dirname):
        os.makedirs(dirname)

# creates a Tk() object
master = Tk()
# sets the geometry of main  
# root window 
master.geometry("200x200")


# function to open a new window  
# on a button click 
def open_new_window():
    # Toplevel object which will  
    # be treated as a new window 
    new_window = Toplevel(master)

    # sets the title of the 
    # Toplevel widget 
    new_window.title("Database entry")

    # sets the geometry of toplevel 
    new_window.geometry("200x200")

    # A Label widget to show in toplevel 
    l0 = Label(new_window, text="Database entry")
    l0.grid(row=0, column=0)
    l1 = Label(new_window, text="License plate")
    l1.grid(row=1, column=0)
    e1 = Entry(new_window)
    e1.grid(row=1, column=1)

    l2 = Label(new_window, text="Name on registration")
    l2.grid(row=2, column=0)
    e2 = Entry(new_window)
    e2.grid(row=2, column=1)

    l3 = Label(new_window, text="Address")
    l3.grid(row=3, column=0)
    e3 = Entry(new_window)
    e3.grid(row=3, column=1)
    btn4 = Button(new_window,
                  text="Click to enter records",
                  command=lambda: enter_records(e1, e2, e3))
    btn4.grid(row=5, column=0)


def enter_records(e1, e2, e3):
    conn = sqlite3.connect(VEHICLES_DB_FILE)
    cursor = conn.cursor()
    cursor.execute("insert into vehicles values('{}','{}','{}')".format(e1.get(), e2.get(), e3.get()))
    conn.commit()
    print("Records added...")
    conn.close()


def run_program():
    detect_video(YOLO(**{"score": 0.7}), "demo_input.mp4", "demo_output.mp4")


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
        cursor.execute("select * from vehicles where plate='{}';".format(license_plate_str))
        result = cursor.fetchone()
        print(result)
        if result is None:
            continue
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


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    label = Label(master,
                  text="This is the main window")
    label.pack(pady=10)
    # a button widget which will open a
    # new window on button click
    btn = Button(master,
                 text="Click to open database entry",
                 command=open_new_window)
    btn.pack(pady=10)
    btn2 = Button(master,
                  text="Click to run helmet detection program",
                  command=run_program)
    btn2.pack(pady=10)

    btn3 = Button(master,
                  text="Click to generate challans",
                  command=generate_challans)
    btn3.pack(pady=10)
    # mainloop, runs infinitely
    mainloop()


if __name__ == "__main__":
    main()
