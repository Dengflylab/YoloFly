from kivy.clock import Clock
from kivy.graphics import Color, Rectangle, Point, GraphicException
from kivy.graphics.texture import Texture
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.lang import Builder
from kivy.lang.builder import Builder
from kivy.properties import StringProperty
from kivy.utils import get_color_from_hex, platform
from kivy.uix.image import Image
from kivy.uix.label import Label

from kivymd.app import MDApp
from kivymd.icon_definitions import md_icons
from kivymd.uix.button import MDFlatButton, MDFloatingActionButton
from kivymd.uix.dialog import MDDialog
from kivymd.uix.floatlayout import MDFloatLayout
from kivymd.uix.list import IRightBodyTouch, OneLineAvatarIconListItem, OneLineListItem, TwoLineAvatarListItem, TwoLineAvatarIconListItem, MDList, IconLeftWidget, IconRightWidget, ImageLeftWidget
from kivymd.uix.selectioncontrol import MDCheckbox

from plyer import filechooser

import cv2, os, json, torch
import numpy as np
import pandas as pd
from utils.augmentations import letterbox
from utils.datasets import LoadImages
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords, xyxy2xywh
from utils.plots import Annotator, colors
from utils.torch_utils import  select_device

AUTO_PATH = os.path.realpath(__file__).replace("main.py", "")

print(AUTO_PATH)

class ListItemWithCheckbox(OneLineAvatarIconListItem):
    '''Custom list item.'''

    icon = StringProperty("bee")

class RightCheckbox(IRightBodyTouch, MDCheckbox):
    '''Custom right container.'''
    active = True

class User_data:
    FlyMatrix = {}
    Add_dict_body = {}
    Add_dict_head = {}
    Add_dict_body_c = {}
    Add_dict_head_c = {}
    select_stat = "fill"
    ck_list = []
    ck_list_c = []
    AAA = "before"
    Face_page = "/home/ken/Videos/4_clips.mp4"
    texture = None
    slider_value = 0
    Vide_mode = "Play"
    #Split = "\"
    def __init__(self):
        capture = cv2.VideoCapture(self.Face_page)
        ret, frame = capture.read()
        if ret:
            self.CV_texture(frame)

    def CV_texture(self, frame):
        buf1 = cv2.flip(frame, 0)
        buf = buf1.tostring()
        image_texture =Texture.create(
            size = (frame.shape[1], frame.shape[0]), colorfmt='bgr'
        )
        image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.texture = image_texture

class Video_show(Image):
    id = "IMAGE"
    Scroll = None
    Touch_proper = "body"
    File = "/mnt/8A26661926660713/Github/yolov5/logo.png"
    N_frame = 0
    def pop(self):
        pass

    def CV_texture(self, frame):
        buf1 = cv2.flip(frame, 0)
        buf = buf1.tobytes()
        image_texture =Texture.create(
            size = (frame.shape[1], frame.shape[0]), colorfmt='bgr'
        )
        image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.texture = image_texture

    def __init__(self, File = File,**kwargs):
        super(Video_show, self).__init__(**kwargs)
        self.capture = cv2.VideoCapture(File)
        ret, self.frame = self.capture.read()
        if ret:
            self.CV_texture(self.frame)

    def update(self, **kwargs):
        self.capture = cv2.VideoCapture(self.File)
        ret, self.frame = self.capture.read()
        if ret:
            self.CV_texture(self.frame)

    def update_frame(self, **kwargs):
        self.capture = cv2.VideoCapture(self.File)
        self.capture.set(1, self.N_frame-1)
        ret, self.frame = self.capture.read()
        if ret:
            self.CV_texture(self.frame)
        return self.frame

    def get_frame(self, **kwargs):
        self.capture = cv2.VideoCapture(self.File)
        self.capture.set(1, self.N_frame-1)
        ret, self.frame = self.capture.read()
        if ret:
            return self.frame

    def on_touch_down(self, touch):
        if not self.collide_point(*touch.pos):
            return False
        win = self.get_parent_window()
        ud = touch.ud
        ud['group'] = g = str(touch.uid)
        pointsize = 5
        if 'pressure' in touch.profile:
           ud['pressure'] = touch.pressure
           pointsize = (touch.pressure * 100000) ** 2
           ud['color'] = random()

        with self.canvas:
            Color(1, 1, 1, .8, mode='rgba', group=g)
            ud['lines'] = [
            Rectangle(pos=(touch.x, 0), size=(1, win.height), group=g),
            Rectangle(pos=(0, touch.y), size=(win.width, 1), group=g),
            Point(points=(touch.x, touch.y), source='particle.png',
                  pointsize=pointsize, group=g)]

        #self.update_touch_label(ud['label'], touch)
        touch.grab(self)
        return True

    def on_touch_up(self, touch):
        if touch.grab_current is not self:
            return
        touch.ungrab(self)
        ud = touch.ud
        self.canvas.remove_group(ud['group'])
        value = (touch.x, touch.y)
        print("I'm here:", touch.x, touch.y)

        ## Add the point to the data
        if self.Touch_proper == "body":
            Num = len(User_Data.Add_dict_body)
            tmp_pos = self.relative_pos(touch.x, touch.y)
            User_Data.Add_dict_body.update({"fly_"+str(Num):{"body": tmp_pos}})
            icons = list(md_icons.keys())
            self.Scroll_Updata()
            print(User_Data.Add_dict_body)

        if self.Touch_proper == "head":
            Num = len(User_Data.Add_dict_head)
            tmp_pos = self.relative_pos(touch.x, touch.y)
            User_Data.Add_dict_head.update({"fly_"+str(Num):{"head": tmp_pos}})
            icons = list(md_icons.keys())
            self.Scroll_Updata()
            print(User_Data.Add_dict_head)

        if self.Touch_proper == "print":
            try:
                self.pop(str(value))
            except:
                pass
    def relative_pos(self, X, Y):
        _L = float(self.Cali_lu.replace(" ", "").split(",")[0])
        _U = float(self.Cali_lu.replace(" ", "").split(",")[1])
        _R = float(self.Cali_rd.replace(" ", "").split(",")[0])
        _D = float(self.Cali_rd.replace(" ", "").split(",")[1])
        _W = _R - _L
        _H = _U - _D
        Re_pos = [(X - _L)/_W, (_U - Y)/_H]
        return Re_pos

    def Scroll_Updata(self):
        frame = np.array(self.frame)
        # when clear, update the frame
        if len(User_Data.Add_dict_head)==0 and len(User_Data.Add_dict_body)==0:
            self.CV_texture(frame)
        else:
            for i in User_Data.Add_dict_head:
                '''
                self.Scroll.add_widget(
                        ListItemWithCheckbox(text=f"{i.replace('fly_', '')}", icon='eye')
                    )
                '''
                try:
                    X = int(User_Data.Add_dict_head[i]['body'][0] * 1920)
                    Y = int(User_Data.Add_dict_head[i]['body'][1] * 1080)
                except:
                    X = int(User_Data.Add_dict_head[i]['head'][0] * 1920)
                    Y = int(User_Data.Add_dict_head[i]['head'][1] * 1080)

                font = cv2.FONT_HERSHEY_SIMPLEX

                frame = cv2.putText(frame, i.replace("fly","H"),(X,Y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

            for ii in User_Data.Add_dict_body:
                '''
                self.Scroll.add_widget(
                        ListItemWithCheckbox(text=f"{ii.replace('fly_', '')}", icon='bee')
                    )
                '''
                X = int(User_Data.Add_dict_body[ii]['body'][0] * 1920)
                Y = int(User_Data.Add_dict_body[ii]['body'][1] * 1080)
                font = cv2.FONT_HERSHEY_SIMPLEX

                frame = cv2.putText(frame, ii,(X,Y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            self.CV_texture(frame)

class Button_fun:
    def __init__(self):
        print("in init")

class MenuScreen(Screen):
    pass

class ProfileScreen(Screen):
    pass

class UploadScreen(Screen):
    pass

# Create the screen manager
sm = ScreenManager()
sm.add_widget(ProfileScreen(name='profile'))
#sm.add_widget(MenuScreen(name='menu'))
sm.add_widget(UploadScreen(name='upload'))

#self.root.get_screen('profile').ids.
class MainApp(MDApp):
    video = False
    dialog = None

    def build(self):
        screen = Builder.load_file("main.kv")
        return screen

    def on_start(self):
        icons = list(md_icons.keys())

        User_Data.ck_list = []
        for i in range(3):
            items = self.scroll_item_add(str(i),"eye")
            self.root.get_screen('profile').ids.scroll.add_widget(items)
        for i in range(3):
            items2 = self.scroll_item_add(str(i),"bee")
            self.root.get_screen('profile').ids.scroll.add_widget(items2)

        # Fly list area
        Video_show.Scroll = self.root.get_screen('profile').ids.scroll

        # Connect button functions
        self.root.get_screen('profile').ids.butn_click_body.on_release = self.butn_click_body
        self.root.get_screen('profile').ids.butn_click_head.on_release = self.butn_click_head
        self.root.get_screen('profile').ids.butn_click_print.on_release = self.butn_click_print
        self.root.get_screen('profile').ids.butn_scroll_refresh.on_release = self.butn_scroll_refresh
        self.root.get_screen('profile').ids.butn_scroll_delete.on_release = self.butn_scroll_delete
        self.root.get_screen('profile').ids.butn_scroll_clear.on_release = self.butn_scroll_clear

        self.root.get_screen('calibrate').ids.butn_scroll_refresh.on_release = self.butn_scroll_refresh_c
        self.root.get_screen('calibrate').ids.butn_scroll_switch.on_release = self.butn_scroll_switch_c
        self.root.get_screen('calibrate').ids.butn_scroll_clear.on_release = self.butn_scroll_clear_c
        self.root.get_screen('calibrate').ids.butn_scroll_remove.on_release = self.butn_scroll_remove_c

        # Video arear
        self.root.get_screen('profile').ids.Video_arear.clear_widgets()
        Video_show.Cali_lu = self.root.get_screen('profile').ids.Cali_lu.text
        Video_show.Cali_rd = self.root.get_screen('profile').ids.Cali_rd.text
        self.Video_show = Video_show()
        self.Video_show.pop = self.show_alert_dialog
        #self.Video_show.bind(on_touch_up=self.On_touch_up)
        self.root.get_screen('profile').ids.Video_arear.add_widget(self.Video_show)

        self.root.get_screen('calibrate').ids.Video_arear.clear_widgets()
        Video_show.Cali_lu = self.root.get_screen('calibrate').ids.Cali_lu.text
        Video_show.Cali_rd = self.root.get_screen('calibrate').ids.Cali_rd.text
        self.Video_show_c = Video_show()
        self.Video_show_c.Touch_proper = "print"
        self.root.get_screen('calibrate').ids.Video_arear.add_widget(self.Video_show_c)

        # Video play
        self.root.get_screen('profile').ids.butn_play_p.on_release = self.butn_play_p
        self.root.get_screen('profile').ids.butn_play_f.on_release = self.butn_play_f
        self.root.get_screen('profile').ids.butn_play_b.on_release = self.butn_play_b

        self.root.get_screen('calibrate').ids.butn_play_p.on_release = self.butn_play_p_c
        self.root.get_screen('calibrate').ids.butn_play_f.on_release = self.butn_play_f_c
        self.root.get_screen('calibrate').ids.butn_play_b.on_release = self.butn_play_b_c

        # Arguments
        self.root.get_screen('profile').ids.butn_vi_choose.on_release = self.choose_vi
        self.root.get_screen('profile').ids.butn_mo_choose.on_release = self.choose_mo
        self.root.get_screen('profile').ids.butn_sa_choose.on_release = self.choose_sa
        self.root.get_screen('profile').ids.butn_start.on_release = self.butn_start

        self.root.get_screen('profile').ids.switch_bh_count.on_release = self.switch_bh_count

        User_Data.AAA = "after"

        self.root.get_screen('profile').ids.slider.bind(value=self.OnSliderValueChange)

        # preload the model
        if self.root.get_screen('profile').ids.choose_model.text!= "":
            device = select_device("")
            try:
                self.model = attempt_load( AUTO_PATH+"/"+self.root.get_screen('profile').ids.choose_model.text, map_location=device)
            except:
                print("Model Not Found")
            #self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.root.get_screen('profile').ids.choose_model.text)

        self.root.get_screen('calibrate').ids.slider.bind(value=self.OnSliderValueChange_c)
        self.root.get_screen('calibrate').ids.butn_pos_inherit_m.on_release = self.butn_pos_inherit_m_c
        self.root.get_screen('calibrate').ids.butn_pos_inherit_p.on_release = self.butn_pos_inherit_p_c
        self.root.get_screen('calibrate').ids.butn_vi_choose.on_release = self.choose_vi_c
        self.root.get_screen('calibrate').ids.butn_sa_caled.on_release = self.butn_sa_caled_c

    def ck_list_clear(self, List_give, Thread = 0):
        List = List_give
        REMOVE = []
        for i in List:
            if List_give.count(i)%2 == Thread:
                REMOVE +=[i]
        for i in REMOVE:
            List.remove(i)
        return List

    def check_press_f(self,val, instance_table):
        User_Data.ck_list.append(val)
        #self.root.get_screen('profile').ids.butn_scroll_delete.ck_list = self.ck_list
        print(User_Data.ck_list)

    def butn_click_body(self):
        self.Video_show.Touch_proper = "body"
    def butn_click_head(self):
        self.Video_show.Touch_proper = "head"
    def butn_click_print(self):
        self.Video_show.Touch_proper = "print"
        #self.Video_show.Touch_proper.pop =

    def OnSliderValueChange(self, instance, value):
        print(int(value))
        if User_Data.Vide_mode == "Play":
            self.Video_show.N_frame = int(value)
            self.Video_show.update_frame()
        Text = "frame: " + str(int(self.root.get_screen('profile').ids.slider.value))
        self.root.get_screen('profile').ids.infor_id.text = Text

    def On_touch_up(self, instance, value):
        if self.Video_show.Touch_proper == "print":
            self.show_alert_dialog(str(value.pos))

    def scroll_item_add(self, i, icon, Active = False):
        items = TwoLineAvatarIconListItem(text=f"{i.replace('fly_','')}"

    )
        right_I = IconRightWidget(icon='nothing.png')
        from functools import partial
        ck = MDCheckbox(active= Active)
        sux = "head"
        if icon == "bee":
            sux = "body"
        ck.bind(on_release=partial(self.check_press_f,i + "," +sux))
        right_I.add_widget(ck)
        items.add_widget(IconLeftWidget(icon=icon))
        items.add_widget(right_I)
        return items

    def butn_scroll_refresh(self):
        Video_show.Scroll.clear_widgets()
        for i in User_Data.Add_dict_head:
            items = self.scroll_item_add(i,"eye")
            self.root.get_screen('profile').ids.scroll.add_widget(items)
        for ii in User_Data.Add_dict_body:
            items = self.scroll_item_add(ii,"bee")
            self.root.get_screen('profile').ids.scroll.add_widget(items)
        print("123")
    def butn_scroll_delete(self):
        print(User_Data.ck_list)
        # find the delete list
        List = self.ck_list_clear(User_Data.ck_list)

        # clear the deletet list
        User_Data.ck_list = []

        # Start to delete storted list
        for i in List:
            fly_id, Dic = i.split(",")
            if Dic == "body":
                User_Data.Add_dict_body.pop(fly_id)
            if Dic == "head":
                User_Data.Add_dict_head.pop(fly_id)

        # Rearrange the storted list after clear
        TMP_b = {}
        Num = -1
        for i in User_Data.Add_dict_body.keys():
            Num += 1
            TMP_b.update({"fly_"+str(Num):User_Data.Add_dict_body[i]})
        TMP_h = {}
        Num = -1
        for i in User_Data.Add_dict_head.keys():
            Num += 1
            TMP_h.update({"fly_"+str(Num):User_Data.Add_dict_head[i]})
        User_Data.Add_dict_body = TMP_b
        User_Data.Add_dict_head = TMP_h
        # refresh the view
        self.butn_scroll_refresh()
        self.Video_show.Scroll_Updata()
    def butn_scroll_clear(self):
        User_Data.Add_dict_body = {}
        User_Data.Add_dict_head = {}
        self.butn_scroll_refresh()
        self.Video_show.Scroll_Updata()

    def choose_vi(self):
        '''
        Call plyer filechooser API to run a filechooser Activity.
        '''
        filechooser.open_file(on_selection=self.handle_selection_vi)

    def handle_selection_vi(self, selection):
        '''
        Callback function for handling the selection response from Activity.
        '''
        self.selec_vi = selection
        self.root.get_screen('profile').ids.choose_video.text = self.selec_vi[0]
        self.cap=cv2.VideoCapture(self.selec_vi[0])
        frame_total = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.root.get_screen('profile').ids.slider.max = frame_total
        # update the image texture
        self.Video_show.File = self.selec_vi[0]
        self.Video_show.update()
        self.video = self.root.get_screen('profile').ids.choose_video.text.split("/")[-1]
        self.video = self.video.split("\\")[-1]
        self.video = self.video.split(".")[0]
        print(self.video)

    def choose_mo(self):
        '''
        Call plyer filechooser API to run a filechooser Activity.
        '''
        filechooser.open_file(on_selection=self.handle_selection_mo)

    def handle_selection_mo(self, selection):
        '''
        Callback function for handling the selection response from Activity.
        '''
        self.selec_mo = selection
        print(self.selec_mo)
        self.root.get_screen('profile').ids.choose_model.text = self.selec_mo[0]

    def choose_sa(self):
        filechooser.open_file(on_selection=self.handle_selection_sa)

    def handle_selection_sa(self, selection):
        self.selec_sa = selection
        print(self.selec_sa)
        self.root.get_screen('profile').ids.choose_save.text = self.selec_sa[0]

    def butn_play_f(self):
        if self.root.get_screen('profile').ids.slider.value < self.root.get_screen('profile').ids.slider.max:
            self.root.get_screen('profile').ids.slider.value += 1
    def butn_play_p(self):
        if self.root.get_screen('profile').ids.butn_play_p.icon == "play":
            try:
                self.cap.set(1, self.root.get_screen('profile').ids.slider.value -1)
                self.root.get_screen('profile').ids.butn_play_p.icon = "pause"
                Clock.schedule_interval(self.video_update, 1.0 / 30)
            except:
                Txt = "file from 'Video': " +   self.root.get_screen('profile').ids.choose_video.text + "is not a video; please choose a video file"
                self.show_alert_dialog(Txt)
        elif self.root.get_screen('profile').ids.butn_play_p.icon == "pause":
            self.root.get_screen('profile').ids.butn_play_p.icon = "play"
            Clock.unschedule(self.video_update)
    def butn_play_b(self):
        if self.root.get_screen('profile').ids.slider.value >= 2 :
            self.root.get_screen('profile').ids.slider.value -= 1

    ## buttons for play

    def video_update(self, dt):
        ret, frame = self.cap.read()
        self.root.get_screen('profile').ids.slider.value = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        if ret:
            self.Video_show.CV_texture(frame)
        else:
            Clock.unschedule(self.video_update)
            self.root.get_screen('profile').ids.slider.value = 1

    def butn_start(self):
        self.switch_bh_count()
        if self.root.get_screen('profile').ids.butn_start.icon == "play":
            try:
                self.cap.set(1, self.root.get_screen('profile').ids.slider.value -1)
                self.root.get_screen('profile').ids.butn_start.icon = "pause"
                self.root.get_screen('profile').ids.butn_start.text = "Stop Annotate"
                self.root.get_screen('profile').ids.butn_start.text_color = 0, 0, 0, 1
                self.root.get_screen('profile').ids.butn_start.md_bg_color = 1, .5, .5, 1
                Clock.schedule_interval(self.video_annotate, 1.0 / 30)
            except:
                Txt = "file from 'Video': " +   self.root.get_screen('profile').ids.choose_video.text + "is not a video; please choose a video file"
                self.show_alert_dialog(Txt)
        elif self.root.get_screen('profile').ids.butn_start.icon == "pause":
            self.butn_start_stop()
            if self.root.get_screen('profile').ids.switch_save.active:
                self.bh_count_csv.close()
                self.bh_count_json.close()

    def butn_start_stop(self):
            self.root.get_screen('profile').ids.butn_start.icon = "play"
            self.root.get_screen('profile').ids.butn_start.text = "Start Anotation"
            self.root.get_screen('profile').ids.butn_start.text_color = 1, 0, 0, 1
            self.root.get_screen('profile').ids.butn_start.md_bg_color = 1, 1, 1, 1
            User_Data.Vide_mode = "Play"
            Clock.unschedule(self.video_annotate)

    def video_annotate(self, dt):
        ret, frame = self.cap.read()
        N_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        if ret:
            self.root.get_screen('profile').ids.slider.value = int(N_frame)
            User_Data.Vide_mode = "Annotate"
            self.Predect(frame)
            # plot the result

        else:
            User_Data.Vide_mode = "Play"
            self.root.get_screen('profile').ids.butn_start.icon = "play"
            self.butn_start_stop()
            self.root.get_screen('profile').ids.slider.value = 1
            if self.root.get_screen('profile').ids.switch_save.active:
                self.bh_count_csv.close()
                self.bh_count_json.close()

    def Predect(self, frame):
        img_size = int(self.root.get_screen('profile').ids.Arg_img_size.text)
        device = select_device("")
        self.model.conf = float(self.root.get_screen('profile').ids.Arg_confidence.text)
        self.model.iou = 0.45
        img = letterbox(frame, img_size, stride = int(self.model.stride.max()), auto=True)[0]
        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        pred = self.model(img, visualize=False)[0]
        pred = non_max_suppression(pred, self.model.conf, self.model.iou, None, False)
        #results = self.model(frame, size=640)
        annotator = Annotator(frame, line_width=3, example=str("names"))
        names = self.model.names
        Clas_dic = {}
        [Clas_dic.update({i:0}) for i in names]
        max_det = int(self.root.get_screen('profile').ids.Arg_target_num.text)
        hide_labels = False
        hide_conf   = False
        #for i in pred[0]:
        for i, det in enumerate(pred):
            gn = torch.tensor(frame.shape)[[1, 0, 1, 0]]
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
            # integer class
            Lable_Result = []
            for *xyxy, conf, cls in det:
                Clas_dic[names[int(cls)]] +=1
                if Clas_dic[names[int(cls)]] <= max_det:
                    if cls<=1 or cls >1 and conf >=  float(self.root.get_screen('profile').ids.Arg_confidence_b.text):
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh)
                        label_tmp = [str(round(line[0].item()))] + [str(i) for i in line[1:]]
                        Lable_Result +=[" ".join(label_tmp)]
                        c = int(cls)
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))

        # update the last line of the behavior count (csv file)
        if self.root.get_screen('profile').ids.switch_save.active:
            Lable_Result_2 = [str(int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)))+" "+i for i in Lable_Result]
            bg_csv = "\n".join(Lable_Result_2)+ "\n"
            self.bh_count_csv.write(bg_csv)

        # update the image texture
        anno_img = annotator.result()
        if anno_img.sum() != 0 :
            self.Video_show.CV_texture(anno_img)
        else:
            self.Video_show.CV_texture(frame)

        # if tracking: tracking functions
        if self.root.get_screen('profile').ids.switch_tarcking.active:
            TB = pd.DataFrame([i.split(" ") for i in Lable_Result]).astype(float)
            TB_head = TB[TB[0]==1]
            TB_head.index = range(len(TB_head.index))
            Num_frame = int(self.root.get_screen('profile').ids.slider.value)
            print("what happen?", Num_frame, self.tar_tr_start)
            if  Num_frame == self.tar_tr_start:
                if len(User_Data.Add_dict_body) == 0:
                    self.FLY_matrix = self.fly_align.TB_dic(TB, Num_frame)
                    ## if head bind
                    if self.root.get_screen('profile').ids.switch_head.active:
                        self.head_bind.main(self.FLY_matrix, Num_frame, TB_head)
                        for fly in self.FLY_matrix[Num_frame].keys():
                            self.FLY_matrix[Num_frame][fly].update({"head":list(TB_head.iloc[int(self.head_bind.MATCH_result[fly]),1:])})
                else:
                    FLY_matrix_tmp = self.fly_align.TB_dic(TB, Num_frame)
                    self.FLY_matrix = self.FLY_matrix_update({Num_frame-1:User_Data.Add_dict_body}, FLY_matrix_tmp, Num_frame)
                    ## head bind, again
                    if self.root.get_screen('profile').ids.switch_head.active and len(User_Data.Add_dict_head) != 0:
                        TB_head = pd.DataFrame([[0] + i["head"] +[0.01, 0.02] for i in list(User_Data.Add_dict_head.values())])
                        self.head_bind.main(self.FLY_matrix, Num_frame, TB_head, 0)
                        try:
                            for fly in self.FLY_matrix[Num_frame].keys():
                                self.FLY_matrix[Num_frame][fly].update({"head":list(TB_head.iloc[int(self.head_bind.MATCH_result[fly]),1:])})
                        except:
                            Txt = "The number of the head and fly is not match"
                            self.show_alert_dialog(Txt)
                            self.butn_start_stop()
                            return

            if Num_frame >  self.tar_tr_start:
                FLY_matrix_tmp = self.fly_align.TB_dic(TB, Num_frame)
                self.FLY_matrix = self.FLY_matrix_update(self.FLY_matrix, FLY_matrix_tmp, Num_frame)
                ## bind the head
                if self.root.get_screen('profile').ids.switch_head.active:
                    self.head_bind.main(self.FLY_matrix, Num_frame, TB_head)
                    for fly in self.FLY_matrix[Num_frame].keys():
                        try:
                            self.FLY_matrix[Num_frame][fly].update({"head":list(TB_head.iloc[int(self.head_bind.MATCH_result[fly]),1:])})
                        except:
                            # Inherate the head from previous frame based on relative position
                            last_body = self.FLY_matrix[Num_frame-1][fly]['body']
                            last_head = self.FLY_matrix[Num_frame-1][fly]['head']
                            new_body  = self.FLY_matrix[Num_frame][fly]['body']
                            rel_pos = [last_head[0] - last_body[0], last_head[1] - last_body[1]]
                            rel_pos_new = [rel_pos[0]+ new_body[0], rel_pos[1]+ new_body[1]]
                            self.FLY_matrix[Num_frame][fly].update({"head": rel_pos_new + last_head[2:4]})

                    #print(FLY_matrix)
            # remove previous frame to release memeory
            if len(self.FLY_matrix) >= 45:
                ID = list(self.FLY_matrix.keys())[0]
                self.FLY_matrix.pop(ID)

            # save the FLY_matrix:
            dic_ID = list(self.FLY_matrix.keys())[-1]
            tmp = {dic_ID:self.FLY_matrix[dic_ID]}
            FLY_matrix_tmp = json.dumps(tmp) + ";"
            # write json object to file
            if self.root.get_screen('profile').ids.switch_save.active:
                self.bh_count_json.write(FLY_matrix_tmp)
            # close file
            #if tracks show:
            if self.root.get_screen('profile').ids.switch_tarck_show.active:
                def f(n,x):
                    a=[0,1,2,3,4,5,6,7,8,9,'A','b','C','D','E','F']
                    b=[]
                    while True:
                        s=n//x
                        y=n%x
                        b=b+[y]
                        if s==0:
                            break
                        n=s
                    b.reverse()
                    Result = [a[i] for i in b]
                    return  Result
                if Num_frame-1 - self.tar_tr_start <= 45:
                    frame_start = self.tar_tr_start
                else:
                    frame_start = Num_frame - 1 -45
                for frame_track in range(list(self.FLY_matrix.keys())[0], Num_frame-1):
                    FLY_id = 0
                    FLY_list = [flys_id for flys_id in list(self.FLY_matrix[Num_frame].keys())]
                    FLY_list.sort()
                    #print(FLY_list)
                    for flys_id in FLY_list:
                        POSITION = (int((self.FLY_matrix[frame_track+1][flys_id]['body'][0] * len(frame[0]))),                       int(self.FLY_matrix[frame_track+1][flys_id]['body'][1] * len(frame)))
                        track_Color = [255, 255, 255]
                        for i_num in f(FLY_id,3):
                            track_Color[i_num]=track_Color[i_num] -50
                        cv2.circle(frame, POSITION, 2, track_Color, thickness=4, lineType=8, shift=0)
                        FLY_id +=1
                        #if head_bind:
                        if self.root.get_screen('profile').ids.switch_head.active:
                            POSITION_p = (int(self.FLY_matrix[Num_frame][flys_id]['body'][0] * len(frame[0])),                       int(self.FLY_matrix[Num_frame][flys_id]['body'][1] * len(frame)))
                            HEAD_p = (int(self.FLY_matrix[Num_frame][flys_id]['head'][0] * len(frame[0])),                       int(self.FLY_matrix[Num_frame][flys_id]['head'][1] * len(frame)))
                            #print(POSITION, HEAD)
                            cv2.arrowedLine(frame, POSITION_p, HEAD_p, (track_Color), 3, tipLength = 0.5)
                self.Video_show.CV_texture(frame)

    def FLY_matrix_update(self, FLY_matrix, FLY_matrix_tmp, frame):
        '''
        As you can see, this function as is its name
        '''
        Threads = float(self.root.get_screen('profile').ids.Arg_leap_t.text)
        MATCH_result = self.fly_align.align(FLY_matrix, FLY_matrix_tmp, frame, Threads)
        FLY_LIST = {v: k for k, v in MATCH_result.items()} # Revers the values and keys
        FLY_LIST = {v: k for k, v in sorted(FLY_LIST.items(), key=lambda item: item[1])} # sort the diction
        Dic_update = {frame:{}}
        for i in FLY_LIST:
            Dic_update[frame].update({i:FLY_matrix_tmp[frame][FLY_LIST[i]] })
        FLY_matrix.update(Dic_update)

        # Searching lost one:
        LOST_ID = list(FLY_matrix[frame-1].keys())
        fly_new = list(FLY_matrix[frame].keys())
        [LOST_ID.remove(i) for i in fly_new]
        if len(LOST_ID) > 0:
            Lost_dic = {}
            for id_old in LOST_ID:
                FLY_matrix[frame].update({id_old:FLY_matrix[frame-1][id_old]})


        return FLY_matrix

    def switch_bh_count(self):
        if self.root.get_screen('profile').ids.switch_bh_count.active:
            if self.video and self.root.get_screen('profile').ids.switch_tarcking.active:
                    self.tar_tr_start = int(self.root.get_screen('profile').ids.slider.value)
                    from utils import Fly_Tra
                    self.fly_align = Fly_Tra.fly_align()
                    from utils.Head_bind import head_match
                    self.head_bind = head_match()
            if self.root.get_screen('profile').ids.switch_save.active:
                Name = self.video+ "_" + str(int(self.root.get_screen('profile').ids.slider.value))
                try:
                    os.remove("csv/" + Name+".csv")
                    os.remove("csv/" + Name+".json")
                except:
                    pass
                self.bh_count_csv = open(self.root.get_screen('profile').ids.choose_save.text+"/" + Name+".csv", "a")
                self.bh_count_json = open(self.root.get_screen('profile').ids.choose_save.text+"/" + Name+".json", "a")

        else:
            self.root.get_screen('profile').ids.switch_tarcking.active = False
            self.root.get_screen('profile').ids.switch_tarck_show.active = False
            self.root.get_screen('profile').ids.switch_head.active = False

    def show_alert_dialog(self, Txt= "???"):
        self.dialog = None
        if not self.dialog:
            self.dialog = MDDialog(text= Txt)
        self.dialog.open()

    ###################################
    # Special funcitons for Calibrate #
    ###################################

    def butn_play_f_c(self):
        D_frame = int(self.root.get_screen('calibrate').ids.Arg_number_N.text)
        if self.root.get_screen('calibrate').ids.slider.value < self.root.get_screen('calibrate').ids.slider.max:
            try:
                self.root.get_screen('calibrate').ids.slider.value += D_frame
            except:
                print("?")
    def butn_play_p_c(self):

        if self.root.get_screen('calibrate').ids.butn_play_p.icon == "play":
            try:
                self.cap_c.set(1, self.root.get_screen('calibrate').ids.slider.value -1)
                self.root.get_screen('calibrate').ids.butn_play_p.icon = "pause"
                Clock.schedule_interval(self.video_update_c, 1.0 / 30)
            except:
                print("no video")
        elif self.root.get_screen('calibrate').ids.butn_play_p.icon == "pause":
            self.root.get_screen('calibrate').ids.butn_play_p.icon = "play"
            Clock.unschedule(self.video_update_c)
    def butn_play_b_c(self):
        D_frame = int(self.root.get_screen('calibrate').ids.Arg_number_N.text)
        try:
            self.root.get_screen('calibrate').ids.slider.value -= D_frame
        except:
            print("?")

    def OnSliderValueChange_c(self, instance, value):
        print(int(value))
        self.Video_show_c.N_frame = int(value)
        frame = self.Video_show_c.get_frame()
        if self.root.get_screen('calibrate').ids.switch_annotate.active:
            self.Annotate_c(frame)
        else:
            self.Video_show_c.CV_texture(frame)
        Text = "frame: " + str(int(self.root.get_screen('calibrate').ids.slider.value))
        self.root.get_screen('calibrate').ids.infor_id.text = Text

    def video_update_c(self, dt):
        ret, frame = self.cap_c.read()
        self.root.get_screen('calibrate').ids.slider.value = int(self.cap_c.get(cv2.CAP_PROP_POS_FRAMES))
        if ret:
            if self.root.get_screen('calibrate').ids.switch_annotate.active:
                self.Annotate_c(frame)
            else:
                print("herer")
                self.Video_show_c.CV_texture(frame)
        else:
            self.root.get_screen('calibrate').ids.butn_play_p.icon = "play"
            Clock.unschedule(self.video_update_c)
            self.root.get_screen('calibrate').ids.slider.value = 1

    def check_press_f_c(self,val, instance_table):
        User_Data.ck_list_c.append(val)
        #self.root.get_screen('profile').ids.butn_scroll_delete.ck_list = self.ck_list
        print(User_Data.ck_list_c)

    def scroll_item_add_c(self, i, icon, Active = False):
        items = TwoLineAvatarIconListItem(text=f"{i.replace('fly_','')}"

    )
        right_I = IconRightWidget(icon='nothing.png')
        from functools import partial
        ck = MDCheckbox(active= Active)
        sux = "head"
        if icon == "bee":
            sux = "body"
        ck.bind(on_release=partial(self.check_press_f_c,i + "," +sux))
        right_I.add_widget(ck)
        items.add_widget(IconLeftWidget(icon=icon))
        items.add_widget(right_I)
        return items

    # buttons for scroll items
    def butn_scroll_refresh_c(self):
        User_Data.select_stat = "full"
        self.read_data_c()
        self.root.get_screen('calibrate').ids.scroll.clear_widgets()
        N_frame = int(self.root.get_screen('calibrate').ids.slider.value)
        try:
            if "head" in self.FLY_matrix_c[str(N_frame)]["fly_0"].keys():
                for i in self.FLY_matrix_c[str(N_frame)].keys():
                    items = self.scroll_item_add_c(i,"eye", True)
                    self.root.get_screen('calibrate').ids.scroll.add_widget(items)
            for i in self.FLY_matrix_c[str(N_frame)].keys():
                items = self.scroll_item_add_c(i,"bee", True)
                self.root.get_screen('calibrate').ids.scroll.add_widget(items)
            User_Data.ck_list_c = []
        except:
            Txt = "Can't load Annotate '.json' file from: "
            self.show_alert_dialog(Txt + self.root.get_screen('calibrate').ids.choose_data.text)
    def butn_scroll_switch_c(self):
        Start_frame = int(self.root.get_screen('calibrate').ids.slider.value)
        try:
            Last_frame = int(list(self.FLY_matrix_c.keys())[-1]) +1
        except:
            Txt = "Please refresh the list first"
            self.show_alert_dialog(Txt)
            return

        LIST = self.ck_list_clear(User_Data.ck_list_c)
        Switch_Blist = [i.split(",")[0] for i in LIST if "body" in i]
        Switch_Hlist = [i.split(",")[0] for i in LIST if "head" in i]
        Txt = "your select is: " + str(Switch_Blist) +" Please make sure that there are only two targets in the list. (PS: Make sure run selection-remove button (dash rectangle with an frok mark) before select)"
        if len(Switch_Blist) != 0 and len(Switch_Blist) != 2:
            self.show_alert_dialog(Txt)
            return
        if len(Switch_Hlist) != 0 and len(Switch_Hlist) != 2:
            self.show_alert_dialog(Txt)
            return
        if len(Switch_Blist) == 0 and len(Switch_Hlist) == 0:
            self.show_alert_dialog(Txt)
            return

        if len(Switch_Blist) == 2:
            for i in range(Start_frame, Last_frame):
                i = str(i)
                TMP = self.FLY_matrix_c[i][Switch_Blist[1]]
                self.FLY_matrix_c[i][Switch_Blist[1]] = self.FLY_matrix_c[i][Switch_Blist[0]]
                self.FLY_matrix_c[i][Switch_Blist[0]] = TMP
        if len(Switch_Hlist) == 2:
            print(Switch_Hlist)
            for i in range(Start_frame, Last_frame):
                i = str(i)
                TMP = self.FLY_matrix_c[i][Switch_Hlist[1]]['head']
                self.FLY_matrix_c[i][Switch_Hlist[1]]['head'] = self.FLY_matrix_c[i][Switch_Hlist[0]]['head']
                self.FLY_matrix_c[i][Switch_Hlist[0]]['head'] = TMP
    def butn_scroll_clear_c(self):
        self.butn_scroll_refresh_c()
        self.root.get_screen('calibrate').ids.scroll.clear_widgets()
        self.Video_show_c.Scroll_Updata()
    def butn_scroll_remove_c(self):
        User_Data.select_stat = "vacum"
        self.read_data_c()
        self.root.get_screen('calibrate').ids.scroll.clear_widgets()
        N_frame = int(self.root.get_screen('calibrate').ids.slider.value)
        try:
            if "head" in self.FLY_matrix_c[str(N_frame)]["fly_0"].keys():
                for i in self.FLY_matrix_c[str(N_frame)].keys():
                    items = self.scroll_item_add_c(i,"eye", False)
                    self.root.get_screen('calibrate').ids.scroll.add_widget(items)
            for i in self.FLY_matrix_c[str(N_frame)].keys():
                items = self.scroll_item_add_c(i,"bee", False)
                self.root.get_screen('calibrate').ids.scroll.add_widget(items)
            User_Data.ck_list_c = []
        except:
            Txt = "Can't load Annotate '.json' file from: "
            self.show_alert_dialog(Txt + self.root.get_screen('calibrate').ids.choose_data.text)

    def choose_vi_c(self):
        '''
        Call plyer filechooser API to run a filechooser Activity.
        '''
        filechooser.open_file(on_selection=self.handle_selection_vi_c)
    def handle_selection_vi_c(self, selection):
        '''
        Callback function for handling the selection response from Activity.
        '''
        self.selec_vi = selection
        self.root.get_screen('calibrate').ids.choose_video.text = self.selec_vi[0]
        self.cap_c=cv2.VideoCapture(self.selec_vi[0])
        frame_total = self.cap_c.get(cv2.CAP_PROP_FRAME_COUNT)
        self.root.get_screen('calibrate').ids.slider.max = frame_total
        # update the image texture
        self.Video_show_c.File = self.selec_vi[0]
        self.Video_show_c.update()
        self.video_c = self.root.get_screen('calibrate').ids.choose_video.text.split("/")[-1]
        self.video_c = self.video_c.split("\\")[-1]
        self.video_c = self.video_c.split(".")[0]
        print(self.video_c)

    def read_data_c(self):
        try:
            F = open(self.root.get_screen('calibrate').ids.choose_data.text).read()
            self.FLY_matrix_c = {}
            [self.FLY_matrix_c.update(json.loads(i)) for i in F.split(";")[:-1]]
            print(self.FLY_matrix_c)
        except:
            Txt = "Can't load Annotate '.json' file from: "
            self.show_alert_dialog(Txt + self.root.get_screen('calibrate').ids.choose_data.text)
    def video_annotate_c(self, dt):
        ret, frame = self.cap_c.read()
        N_frame = self.cap_c.get(cv2.CAP_PROP_POS_FRAMES)
        if ret:
            self.root.get_screen('calibrate').ids.slider.value = int(N_frame)
            User_Data.Vide_mode = "Annotate"
            self.Annotate_c(frame)
            # plot the result

        else:
            User_Data.Vide_mode = "Play"
            Clock.unschedule(self.video_annotate_c)
            self.root.get_screen('calibrate').ids.slider.value = 1


    def Annotate_c(self, frame):
        LIST = self.ck_list_clear(User_Data.ck_list_c)
            #print(LIST)
        Non_list = [i.split(",")[0] for i in LIST if "body" in i]

        def f(n,x):
            '''
            Convert numbers
            '''
            a=[0,1,2,3,4,5,6,7,8,9,'A','b','C','D','E','F']
            b=[]
            while True:
                s=n//x
                y=n%x
                b=b+[y]
                if s==0:
                    break
                n=s
            b.reverse()
            Result = [a[i] for i in b]
            return  Result

        Num_frame = int(self.root.get_screen('calibrate').ids.slider.value)
        try:
            TMP = self.FLY_matrix_c[str(Num_frame)]
            print(TMP)
        except:
            Txt = "Can't load Annotate '.json' file from: "
            self.show_alert_dialog(Txt + self.root.get_screen('calibrate').ids.choose_data.text)
            return None
        # if id show
        if self.root.get_screen('calibrate').ids.switch_id_show.active:
            for fly_id in TMP.keys():
                if User_Data.select_stat == "full" and fly_id not in Non_list or User_Data.select_stat == "vacum" and fly_id in Non_list :
                    BODY = TMP[fly_id]['body']
                    tc_label_x =  int(BODY[0] * len(frame[0]))
                    tc_label_y =  int(BODY[1] * len(frame))
                    frame = cv2.putText(frame, fly_id ,(tc_label_x, tc_label_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (102, 205, 0), 3)
        FLY_list = [flys_id for flys_id in list(self.FLY_matrix_c[str(Num_frame)].keys())]
        FLY_list.sort()
        # if tracks show
        if self.root.get_screen('calibrate').ids.switch_tarck_show.active:
            Start_F = int(self.root.get_screen('calibrate').ids.Arg_track_frame.text)
            print("1:", Start_F)
            if Start_F > Num_frame:
                Start_F = int(list(self.FLY_matrix_c.keys())[0])
            else:
                Start_F = Num_frame - Start_F
            print("2:", Start_F)
            for frame_track in range(Start_F, Num_frame-1):
                FLY_id = 0
                for flys_id in FLY_list:
                    if User_Data.select_stat == "full" and flys_id not in Non_list or User_Data.select_stat == "vacum" and flys_id in Non_list :
                        POSITION = (int((self.FLY_matrix_c[str(frame_track+1)][flys_id]['body'][0] * len(frame[0]))),                       int(self.FLY_matrix_c[str(frame_track+1)][flys_id]['body'][1] * len(frame)))
                        track_Color = [255, 255, 255]
                        for i_num in f(FLY_id,3):
                            track_Color[i_num]=track_Color[i_num] -50
                        frame = cv2.circle(frame, POSITION, 2, track_Color, thickness=4, lineType=8, shift=0)
                        FLY_id +=1
        #if head_bind:
        if self.root.get_screen('calibrate').ids.switch_head.active:
            track_Color = [255, 255, 255]
            FLY_id = 0
            Num_frame = str(Num_frame)
            for flys_id in FLY_list:
                if User_Data.select_stat == "full" and flys_id not in Non_list or User_Data.select_stat == "vacum" and flys_id in Non_list :
                    for i_num in f(FLY_id,3):
                        track_Color[i_num]=track_Color[i_num] -50
                    FLY_id +=1
                    POSITION_p = (int(self.FLY_matrix_c[Num_frame][flys_id]['body'][0] * len(frame[0])),                       int(self.FLY_matrix_c[Num_frame][flys_id]['body'][1] * len(frame)))
                    HEAD_p = (int(self.FLY_matrix_c[Num_frame][flys_id]['head'][0] * len(frame[0])),                       int(self.FLY_matrix_c[Num_frame][flys_id]['head'][1] * len(frame)))
                    #print(POSITION, HEAD)
                    cv2.arrowedLine(frame, POSITION_p, HEAD_p, (track_Color), 3, tipLength = 0.5)

        self.Video_show_c.CV_texture(frame)

    def butn_pos_inherit_m_c(self):
        N_frame = int(self.root.get_screen('calibrate').ids.slider.value)
        D_frame = int(self.root.get_screen('calibrate').ids.Arg_number_N.text)
        LIST = self.ck_list_clear(User_Data.ck_list_c)
        print(LIST)
        if N_frame + D_frame > int(self.root.get_screen('calibrate').ids.slider.max):
            print(N_frame + D_frame)
            Txt = "Please chech the current frame and the value of the 'N='"
            self.show_alert_dialog(Txt)
            return
        if N_frame == 1:
            self.show_alert_dialog("Can't inherit the data from frame 0, please make sure that the position of the slider")
            return
        if User_Data.select_stat != "vacum":
            Txt = "Make sure run selection-remove button (dash rectangle with an frok mark) before select"
            self.show_alert_dialog(Txt)
            return
        Switch_Blist = [i.split(",")[0] for i in LIST if "body" in i]
        Switch_Hlist = [i.split(",")[0] for i in LIST if "head" in i]
        for NN in range(N_frame , N_frame + D_frame):
            if len(Switch_Blist) > 0:
                for fly_id in Switch_Blist:
                    self.FLY_matrix_c[str(NN)][fly_id] = self.FLY_matrix_c[str(NN -1 )][fly_id]
            if len(Switch_Blist) == 0 and len(Switch_Hlist) > 0:
                for fly_id in Switch_Hlist:
                    self.FLY_matrix_c[str(NN)][fly_id]['head'] = self.FLY_matrix_c[str(NN -1 )][fly_id]['head']
    def butn_pos_inherit_p_c(self):
        N_frame = int(self.root.get_screen('calibrate').ids.slider.value)
        D_frame = int(self.root.get_screen('calibrate').ids.Arg_number_N.text)
        LIST = self.ck_list_clear(User_Data.ck_list_c)
        print(LIST)
        if N_frame - D_frame <= 0:
            print(N_frame + D_frame)
            Txt = "Please chech the current frame and the value of the 'N=' (Error code: 'N_frame - D_frame > 0:')"
            self.show_alert_dialog(Txt)
            return
        if N_frame == int(self.root.get_screen('calibrate').ids.slider.max):
            self.show_alert_dialog("Can't inherit the data from the last frame, please make sure that the position of the slider (Code Error: 'N_frame == slider.max')")
            return
        if User_Data.select_stat != "vacum":
            Txt = "Make sure run selection-remove button (dash rectangle with an frok mark) before select"
            self.show_alert_dialog(Txt)
            return
        Switch_Blist = [i.split(",")[0] for i in LIST if "body" in i]
        Switch_Hlist = [i.split(",")[0] for i in LIST if "head" in i]

        for NN in range(N_frame - D_frame+1, N_frame+1):
            if len(Switch_Blist) > 0:
                for fly_id in Switch_Blist:
                    self.FLY_matrix_c[str(NN)][fly_id] = self.FLY_matrix_c[str(N_frame +1 )][fly_id]

            if len(Switch_Blist) == 0 and len(Switch_Hlist) > 0:
                for fly_id in Switch_Hlist:
                    self.FLY_matrix_c[str(NN)][fly_id]['head'] = self.FLY_matrix_c[str(N_frame +1 )][fly_id]['head']
    def butn_sa_caled_c(self):
        print(type(self.video_c))
        print(self.video_c)
        print(str(self.video_c))
        Name = self.video_c +"_caled.json"
        try:
            os.remove("csv/" + Name)
        except:
            pass
        FF = open(self.root.get_screen('profile').ids.choose_save.text+"/" + Name, "a")
        try:
            len(self.FLY_matrix_c)
        except:
            Txt = "Please load the fly list first"
            self.show_alert_dialog(Txt)
            return
        for dic_ID in self.FLY_matrix_c.keys():
            tmp = {dic_ID:self.FLY_matrix_c[dic_ID]}
            FLY_matrix_tmp = json.dumps(tmp) + ";"
            FF.write(FLY_matrix_tmp)
        FF.close()

User_Data = User_data()

MainApp().run()
