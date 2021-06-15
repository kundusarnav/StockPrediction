from functools import partial
import os
import tkinter as tk
from tkinter import ttk
def get_selected_file_name(file_menu):
    filename = file_menu.get()
    os.system(filename)
folder = os.path.realpath(r'C:\Users\Sanjib Kundu\PycharmProjects\StockPrediction')
filelist = [fname for fname in os.listdir(folder)]
master = tk.Tk()
master.geometry('1200x800')
master.title('Select file')
optmenu = ttk.Combobox(master, values=filelist, state='readonly')
optmenu.pack(fill='x')
button_select = tk.Button(master,text="Read File",width=20,height=7,compound=tk.CENTER,command=partial(get_selected_file_name, optmenu))
button_select.pack(side=tk.RIGHT)
master.mainloop()