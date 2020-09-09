import sqlite3
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import datetime
import os
from tkinter import * 
from tkinter.ttk import *
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas as canvas1
import datetime
# creates a Tk() object 
master = Tk() 
  
# sets the geometry of main  
# root window 
master.geometry("200x200") 
  
  
# function to open a new window  
# on a button click 
def openNewWindow(): 
      
    # Toplevel object which will  
    # be treated as a new window 
    newWindow = Toplevel(master) 
  
    # sets the title of the 
    # Toplevel widget 
    newWindow.title("Database entry") 
  
    # sets the geometry of toplevel 
    newWindow.geometry("200x200") 
  
    # A Label widget to show in toplevel 
    l0 = Label(newWindow, text ="Database entry")
    l0.grid(row=0,column=0)
    l1 = Label(newWindow, text ="License plate")
    l1.grid(row=1,column=0)
    e1 = Entry(newWindow)
    e1.grid(row=1,column=1)
    
    l2 = Label(newWindow, text ="Name on registration")
    l2.grid(row=2,column=0)
    e2 = Entry(newWindow)
    e2.grid(row=2,column=1)

    l3 = Label(newWindow, text ="Address")
    l3.grid(row=3,column=0)
    e3 = Entry(newWindow)
    e3.grid(row=3,column=1)
    btn4 = Button(newWindow,  
             text ="Click to enter records",  
             command = lambda : enter_records(e1,e2,e3)) 
    btn4.grid(row=5,column=0)
          
          
          
def enter_records(e1,e2,e3):
    conn = sqlite3.connect('vehicles.db')

    cursor = conn.cursor()

    cursor.execute("insert into vehicles values('{}','{}','{}')".format(e1.get(),e2.get(),e3.get()))
    

    conn.commit()
    print("Records added...")
    conn.close()

          
    
def run_program():
  
    os.chdir('/home/sriram/dad/cisco_project_3sep/keras-yolo3/')
    os.system('python3 yolo_video.py --score 0.7 --input demo_input.mp4 --output demo_output.mp4')
    
            
def generate_challans():
    #gj1md1577
    #gj01uk5541
    os.chdir('/home/sriram/dad/cisco_project_3sep/keras-yolo3/')
    os.system('python3 lpr.py')
    for f in os.listdir('violations'):
        lplate = f.split('.')[0]
        conn = sqlite3.connect('../vehicles.db')
        cursor = conn.cursor()
        cursor.execute("select * from vehicles where plate='{}';".format(lplate))
        result = cursor.fetchone();
        print(result)
        if result is None:
            continue
        name = result[1]
        address = result[2]

        conn.close()
        
    
        canvas = canvas1.Canvas(os.path.join('challans',lplate+".pdf"), pagesize=letter)
        canvas.setLineWidth(.3)
        canvas.setFont('Helvetica', 12)

        canvas.drawString(30,750,'CHALLAN')
        canvas.drawString(30,730,'TRAFFIC POLICE DEPARTMENT, BENGALURU')
        canvas.drawString(30,710,str(datetime.datetime.now()))
        canvas.line(10,707,580,707)

        canvas.drawString(30,650,'VIOLATION:')
        canvas.drawString(300,650,"NOT WEARING HELMET")


        canvas.drawString(30,600,'LICENSE PLATE:')
        canvas.drawString(300,600,lplate)

        canvas.drawString(30,550,'NAME:')
        canvas.drawString(300,550,name)

        canvas.drawString(30,500,'ADDRESS:')
        canvas.drawString(300,500,address)
        
        canvas.drawString(30,450,'FINE:')
        canvas.drawString(300,450,'Rs. 500')

        canvas.drawImage(os.path.join('violations',f), 30,150,250,250,preserveAspectRatio=True, mask='auto')


        canvas.save()
    
    labelr = Label(master,  
              text ="All challans generated, stored in folder /home/sriram/dad/cisco_project_3sep/keras-yolo3/challans") 
  
    labelr.pack(pady = 10) 

    #labelr['text'] = ""
    

  
  
label = Label(master,  
              text ="This is the main window") 
  
label.pack(pady = 10) 
  
# a button widget which will open a  
# new window on button click 
btn = Button(master,  
             text ="Click to open database entry",  
             command = openNewWindow) 
btn.pack(pady = 10) 


btn2 = Button(master,  
             text ="Click to run helmet detection program",  
             command = run_program) 
btn2.pack(pady = 10) 
  
  
btn3 = Button(master,  
             text ="Click to generate challans",  
             command = generate_challans) 
btn3.pack(pady = 10) 

# mainloop, runs infinitely 
mainloop() 
