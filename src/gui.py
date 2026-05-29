import threading
import tkinter as tk
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from routing import route_between


class TBRGSApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Traffic-Based Route Guidance System")
        self.root.geometry("680x560")

        tk.Label(
            root,
            text="Traffic-Based Route Guidance System",
            font=("Calibri", 17, "bold")
        ).pack(pady=12)

        tk.Label(root, text="Origin SCATS Number").pack()
        self.origin_entry = tk.Entry(root, width=30)
        self.origin_entry.pack(pady=2)
        self.origin_entry.insert(0, "2000")

        tk.Label(root, text="Destination SCATS Number").pack()
        self.destination_entry = tk.Entry(root, width=30)
        self.destination_entry.pack(pady=2)
        self.destination_entry.insert(0, "3002")

        tk.Label(root, text="Date and Time  (e.g. 2006-10-18 08:15)").pack()
        self.time_entry = tk.Entry(root, width=30)
        self.time_entry.pack(pady=2)
        self.time_entry.insert(0, "2006-10-18 08:15")

        tk.Label(root, text="Number of routes").pack()
        self.k_entry = tk.Entry(root, width=30)
        self.k_entry.pack(pady=2)
        self.k_entry.insert(0, "5")

        tk.Label(root, text="Algorithm").pack()
        self.algorithm_var = tk.StringVar(value="A* / Part A")
        ttk.Combobox(
            root,
            textvariable=self.algorithm_var,
            values=["A* / Part A", "NetworkX / Backup"],
            state="readonly",
            width=27
        ).pack(pady=2)

        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)

        self.search_btn = tk.Button(
            btn_frame,
            text="Find Routes",
            command=self.find_routes,
            width=18
        )
        self.search_btn.pack(side="left", padx=6)

        tk.Button(
            btn_frame,
            text="Clear",
            command=self.clear_inputs,
            width=18
        ).pack(side="left", padx=6)

        self.results_box = ScrolledText(root, width=78, height=12, font=("Courier", 10))
        self.results_box.pack(pady=8, padx=10)

    def find_routes(self):
        if self.origin_entry.get().strip() == self.destination_entry.get().strip():
            messagebox.showerror("Input Error", "Origin and destination must be different.")
            return

        self.search_btn.configure(state="disabled", text="Searching...")
        self.results_box.delete("1.0", tk.END)
        self.results_box.insert(tk.END, "Finding routes, please wait...\n")

        threading.Thread(target=self.run_search, daemon=True).start()

    def run_search(self):
        try:
            origin = int(self.origin_entry.get())
            destination = int(self.destination_entry.get())
            selected_time = self.time_entry.get()
            k = int(self.k_entry.get())

            routes = route_between(origin, destination, selected_time, k)

            self.root.after(0, self.show_results, routes)

        except ValueError:
            self.root.after(0, lambda: messagebox.showerror(
                "Input Error",
                "Please type valid integers for origin, destination, and number of routes."
            ))
            self.root.after(0, self.reset_button)

        except Exception as error:
            message = str(error)
            self.root.after(0, lambda: messagebox.showerror("Error", message))
            self.root.after(0, self.reset_button)

    def show_results(self, routes):
        self.results_box.delete("1.0", tk.END)

        if not routes:
            self.results_box.insert(tk.END, "No routes found.")
            self.reset_button()
            return

        self.results_box.insert(tk.END, f"Algorithm: {self.algorithm_var.get()}\n\n")

        for number, route in enumerate(routes, start=1):
            self.results_box.insert(
                tk.END,
                f"Route {number}\n"
                f"Path: {route['route']}\n"
                f"Estimated travel time: {route['estimated_minutes']} minutes\n\n"
            )

        self.reset_button()

    def clear_inputs(self):
        self.origin_entry.delete(0, tk.END)
        self.origin_entry.insert(0, "2000")

        self.destination_entry.delete(0, tk.END)
        self.destination_entry.insert(0, "3002")

        self.time_entry.delete(0, tk.END)
        self.time_entry.insert(0, "2006-10-18 08:15")

        self.k_entry.delete(0, tk.END)
        self.k_entry.insert(0, "5")

        self.results_box.delete("1.0", tk.END)

    def reset_button(self):
        self.search_btn.configure(state="normal", text="Find Routes")


def run_gui():
    root = tk.Tk()
    TBRGSApp(root)
    root.mainloop()


if __name__ == "__main__":
    run_gui()