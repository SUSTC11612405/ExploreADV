import tkinter as tk
from PIL import ImageTk
import numpy as np
import torchvision.transforms as T


class MousePositionTracker(tk.Frame):
    """ Tkinter Canvas mouse position widget. """

    def __init__(self, canvas):
        self.canvas = canvas
        self.canv_width = int(self.canvas.cget('width'))
        self.canv_height = int(self.canvas.cget('height'))
        self.reset()

        # Create canvas cross-hair lines.
        xhair_opts = dict(dash=(3, 2), fill='white', state=tk.HIDDEN)
        self.lines = (self.canvas.create_line(0, 0, 0, self.canv_height, **xhair_opts),
                      self.canvas.create_line(0, 0, self.canv_width,  0, **xhair_opts))

    def cur_selection(self):
        return (self.start, self.end)

    def begin(self, event):
        self.hide()
        self.start = (event.x, event.y)  # Remember position (no drawing).

    def update(self, event):
        event.x = max(0, min(event.x, self.canv_width))
        event.y = max(0, min(event.y, self.canv_height))
        self.end = (event.x, event.y)
        self._update(event)
        self._command(self.start, (event.x, event.y))  # User callback.

    def _update(self, event):
        # Update cross-hair lines.
        # print(event.x, event.y)
        self.canvas.coords(self.lines[0], event.x, 0, event.x, self.canv_height)
        self.canvas.coords(self.lines[1], 0, event.y, self.canv_width, event.y)
        self.show()

    def reset(self):
        self.start = (0, 0)
        self.end = (self.canv_width, self.canv_height)

    def hide(self):
        self.canvas.itemconfigure(self.lines[0], state=tk.HIDDEN)
        self.canvas.itemconfigure(self.lines[1], state=tk.HIDDEN)

    def show(self):
        self.canvas.itemconfigure(self.lines[0], state=tk.NORMAL)
        self.canvas.itemconfigure(self.lines[1], state=tk.NORMAL)

    def autodraw(self, command=lambda *args: None):
        """Setup automatic drawing; supports command option"""
        self.reset()
        self._command = command
        self.canvas.bind("<Button-1>", self.begin)
        self.canvas.bind("<B1-Motion>", self.update)
        self.canvas.bind("<ButtonRelease-1>", self.quit)

    def quit(self, event):
        # print(self.cur_selection())
        self.hide()  # Hide cross-hairs.
        # self.reset()


class RectObject:
    """ Widget to display a rectangular area on given canvas defined by two points
        representing its diagonal.
    """
    def __init__(self, canvas, select_opts):
        # Create attributes needed to display selection.
        self.canvas = canvas
        self.select_opts = select_opts

        # Options for areas outside rectanglar selection.
        select_opts = self.select_opts.copy()  # Avoid modifying passed argument.
        select_opts.update(state=tk.HIDDEN)  # Hide initially.

        # Initial extrema of inner and outer rectangles.
        imin_x, imin_y,  imax_x, imax_y = 0, 0,  1, 1

        self.rect = self.canvas.create_rectangle(imin_x, imin_y,  imax_x, imax_y, **select_opts)

    def update(self, start, end):
        # Current extrema of inner and outer rectangles.
        imin_x, imin_y,  imax_x, imax_y = self._get_coords(start, end)

        # Update coords of all rectangles based on these extrema.
        self.canvas.coords(self.rect, imin_x, imin_y,  imax_x, imax_y),

        # Make sure all are now visible.
        self.canvas.itemconfigure(self.rect, state=tk.NORMAL)

    def _get_coords(self, start, end):
        """ Determine coords of a polygon defined by the start and
            end points one of the diagonals of a rectangular area.
        """
        return (min((start[0], end[0])), min((start[1], end[1])),
                max((start[0], end[0])), max((start[1], end[1])))

    def clear(self):
        self.canvas.delete(self.rect)


class BrushObject:
    """ Widget to display a rectangular area on given canvas defined by two points
        representing its diagonal.
    """
    def __init__(self, canvas, H, W, brush_size, resize, select_opts):
        # Create attributes needed to display selection.
        self.canvas = canvas
        self.select_opts = select_opts
        self.width = self.canvas.cget('width')
        self.height = self.canvas.cget('height')
        self.resize = resize
        self.brush_size = brush_size

        # Options for areas outside rectanglar selection.
        select_opts = self.select_opts.copy()  # Avoid modifying passed argument.
        select_opts.update(state=tk.HIDDEN)  # Hide initially.

        self.H = H
        self.W = W

        self.mask = np.zeros((H, W))

        self.grids = []
        for j in range(H):
            col = []
            for i in range(W):
                col.append(self.canvas.create_rectangle(i*resize, j*resize,  (i+1)*resize, (j+1)*resize, **select_opts))
            self.grids.append(col)

    def update(self, start, end):
        # Current extrema of inner and outer rectangles.
        end_x, end_y = end[0] // self.resize, end[1] // self.resize
        for i in range(end_x-self.brush_size+1, end_x+self.brush_size):
            for j in range(end_y-self.brush_size+1, end_y+self.brush_size):
                if 0 <= i < self.W and 0 <= j < self.H:
                    self.mask[j, i] = 1
                    self.canvas.itemconfigure(self.grids[j][i], state=tk.NORMAL)

    def changeW(self, e):
        self.brush_size = e

    def get_mask(self):
        return self.mask

    def clear(self):
        for row in range(self.H):  # Make sure all are now visible.
            for col in range(self.W):
                self.canvas.delete(self.grids[col][row])


def process_image_tensor(tensor, width, height):
    transform = T.ToPILImage()
    image = transform(tensor)
    w, h = image.size
    resize = min(width // w, height // h)
    resize_image = image.resize((w * resize, h * resize))
    return resize_image, resize


class Application(tk.Frame):

    # Default selection object options.
    SELECT_OPTS = dict(dash=(2, 2), stipple='gray25', fill='red',
                          outline='')

    def __init__(self, parent, tensor, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        self.brush_size = 1

        self.rectangle_button = tk.Button(parent, text='Rectangle', command=self.use_rectangle, relief=tk.SUNKEN)
        self.rectangle_button.grid(row=0, column=0)

        self.brush_button = tk.Button(parent, text='Brush', command=self.use_brush)
        self.brush_button.grid(row=0, column=1)

        self.slider = tk.Scale(parent, from_=1, to=5, command=self.changeW, orient=tk.HORIZONTAL)
        self.slider.grid(row=0, column=2)

        self.confirm_button = tk.Button(parent, text='Confirm', command=self.confirm)
        self.confirm_button.grid(row=0, column=3)

        self.active_button = self.rectangle_button

        w, h = 500, 500
        data = tensor.numpy()
        self.region_mask = np.zeros_like(data)

        self.images = [process_image_tensor(tensor[i], w, h) for i in range(tensor.shape[0])]
        self.index = 0

        resize_image, self.resize = self.images[self.index]
        img = ImageTk.PhotoImage(resize_image)

        self.canvas = tk.Canvas(parent, width=img.width(), height=img.height(),
                                borderwidth=0, highlightthickness=0)
        self.canvas.grid(row=1, columnspan=4)
        # self.canvas.pack(expand=True)

        self.canvas.create_image(0, 0, image=img, anchor=tk.NW)
        self.canvas.img = img  # Keep reference.

        self.new_selection_obj()
        self.new_posn_tracker()

    def use_rectangle(self):
        self.activate_button(self.rectangle_button)

    def use_brush(self):
        self.activate_button(self.brush_button)

    def activate_button(self, some_button):
        self.active_button.config(relief=tk.RAISED)
        some_button.config(relief=tk.SUNKEN)
        self.active_button = some_button
        if self.active_button == self.rectangle_button:
            self.slider.configure(state='disabled')
        elif self.active_button == self.brush_button:
            self.slider.configure(state='active')
        self.switch()

    def switch(self):
        self.selection_obj.clear()
        self.new_selection_obj()

    def changeW(self, e):  # change Width of pen through slider
        self.brush_size = int(e)
        self.selection_obj.changeW(self.brush_size)

    def new_selection_obj(self):
        # Create selection object.
        if self.active_button == self.rectangle_button:
            self.selection_obj = RectObject(self.canvas, self.SELECT_OPTS)
        elif self.active_button == self.brush_button:
            H, W = self.region_mask[self.index].shape[1:]
            self.selection_obj = BrushObject(self.canvas, H, W, self.brush_size, self.resize, self.SELECT_OPTS)

    def new_posn_tracker(self):
        # Create mouse position tracker that uses the function.
        self.posn_tracker = MousePositionTracker(self.canvas)
        self.posn_tracker.autodraw(command=self.on_drag)  # Enable callbacks.

    def get_region_mask(self):
        return self.region_mask

    def confirm(self):
        self.update_region_masks()
        self.index += 1
        if self.index >= len(self.images):
            self.quit()
        else:
            self.update()

    def update_region_masks(self):
        if self.active_button == self.rectangle_button:
            start, end = self.posn_tracker.cur_selection()
            start_y = min(start[0], end[0]) // self.resize
            start_x = min(start[1], end[1]) // self.resize
            end_y = max(start[0], end[0]) // self.resize
            end_x = max(start[1], end[1]) // self.resize
            self.region_mask[self.index, :, start_x: end_x, start_y: end_y] = 1.0
        elif self.active_button == self.brush_button:
            mask = self.selection_obj.get_mask()
            self.region_mask[self.index, :] = mask

    def update(self):
        resize_image, self.resize = self.images[self.index]
        img = ImageTk.PhotoImage(resize_image)
        self.canvas.delete('all')
        self.canvas.create_image(0, 0, image=img, anchor=tk.NW)
        self.canvas.img = img  # Keep reference.
        self.new_selection_obj()
        self.new_posn_tracker()

    def on_drag(self, start, end, **kwarg):  # Must accept these arguments.
        # Callback function to update it given two points of its diagonal.
        self.selection_obj.update(start, end)


def region_selector(tensor):

    TITLE = 'Select the region you want to perturb'

    root = tk.Tk()
    root.title(TITLE)

    app = Application(root, tensor)
    app.mainloop()

    mask = app.get_region_mask()
    root.destroy()

    return mask