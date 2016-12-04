import cv2
import numpy
import math
import time
from enum import Enum

cam = cv2.VideoCapture()

def main():
    print ("started")
    #cam.open(0)
    #_, cap = cam.read()
    x = Pipeline()
    #x.set_source0(cap)
    x.set_source0(cv2.imread("/Users/Joseph/Desktop/tower.png"))
    while(True):
        #_, cap = cam.read()
        x.process()
        try:
            print(x.calcCX(x.filter_contours_output))
        except Exception as e:
            print (e)
        time.sleep(1)

class Pipeline:
    """This is a generated class from GRIP.
    To use the pipeline first create a Pipeline instance and set the sources,
    next call the process method,
    finally get and use the outputs.
    """

    def __init__(self):
        """initializes all values to presets or None if need to be set
        """
        self.__source0 = None
        self.__cv_resize_src = self.__source0
        self.__cv_resize_dsize = (0, 0)
        self.__cv_resize_fx = 0.25
        self.__cv_resize_fy = 0.25
        self.__cv_resize_interpolation = cv2.INTER_LINEAR
        self.cv_resize_output = None

        self.__hsv_threshold_input = self.cv_resize_output
        self.__hsv_threshold_hue = [42.0, 123.0]
        self.__hsv_threshold_saturation = [142.0, 255.0]
        self.__hsv_threshold_value = [103.0, 216.0]
        self.hsv_threshold_output = None

        self.__find_contours_input = self.hsv_threshold_output
        self.__find_contours_external_only = False
        self.find_contours_output = None

        self.__filter_contours_contours = self.find_contours_output
        self.__filter_contours_min_area = 50
        self.__filter_contours_min_perimeter = 0
        self.__filter_contours_min_width = 0
        self.__filter_contours_max_width = 1000
        self.__filter_contours_min_height = 0
        self.__filter_contours_max_height = 1000
        self.__filter_contours_solidity = [0, 100]
        self.__filter_contours_max_vertices = 1000000
        self.__filter_contours_min_vertices = 0
        self.__filter_contours_min_ratio = 0
        self.__filter_contours_max_ratio = 1000
        self.filter_contours_output = None


    def process(self):
        """Runs the pipeline.
        Sets outputs to new values.
        Requires all sources to be set.
        """
        #Step CV_resize0:
        self.__cv_resize_src = self.__source0
        (self.cv_resize_output ) = self.__cv_resize(self.__cv_resize_src, self.__cv_resize_dsize, self.__cv_resize_fx, self.__cv_resize_fy, self.__cv_resize_interpolation)
        print("resized")
        #Step HSV_Threshold0:
        self.__hsv_threshold_input = self.cv_resize_output
        (self.hsv_threshold_output ) = self.__hsv_threshold(self.__hsv_threshold_input, self.__hsv_threshold_hue, self.__hsv_threshold_saturation, self.__hsv_threshold_value)
        print("thresholded")
        cv2.imshow("test", self.cv_resize_output)
        try:
        #Step Find_Contours0:
            self.__find_contours_input = self.hsv_threshold_output
            (self.find_contours_output ) = self.__find_contours(self.__find_contours_input, self.__find_contours_external_only)
            print("contours found")
            cv2.imshow("hsv", self.hsv_threshold_output)
            #Step Filter_Contours0:
            self.__filter_contours_contours = self.find_contours_output
            (self.filter_contours_output ) = self.__filter_contours(self.__filter_contours_contours, self.__filter_contours_min_area, self.__filter_contours_min_perimeter, self.__filter_contours_min_width, self.__filter_contours_max_width, self.__filter_contours_min_height, self.__filter_contours_max_height, self.__filter_contours_solidity, self.__filter_contours_max_vertices, self.__filter_contours_min_vertices, self.__filter_contours_min_ratio, self.__filter_contours_max_ratio)
            print("filtered contours")
        except Exception as e:
            print (e)
    def set_source0(self, value):
        """Sets source0 to given value checking for correct type.
        """
        assert isinstance(value, numpy.ndarray) , "Source must be of type numpy.ndarray"
        self.__source0 = value
        print("video source set")


    @staticmethod
    def __cv_resize(src, d_size, fx, fy, interpolation):
        """Resizes an Image.
        Args:
            src: A numpy.ndarray.
            d_size: Size to set the image.
            fx: The scale factor for the x.
            fy: The scale factor for the y.
            interpolation: Opencv enum for the type of interpolation.
        Returns:
            A resized numpy.ndarray.
        """
        return cv2.resize(src, d_size, fx=fx, fy=fy, interpolation=interpolation)

    @staticmethod
    def __hsv_threshold(input, hue, sat, val):
        """Segment an image based on hue, saturation, and value ranges.
        Args:
            input: A BGR numpy.ndarray.
            hue: A list of two numbers the are the min and max hue.
            sat: A list of two numbers the are the min and max saturation.
            lum: A list of two numbers the are the min and max value.
        Returns:
            A black and white numpy.ndarray.
        """
        out = cv2.cvtColor(input, cv2.COLOR_BGR2HSV)
        return cv2.inRange(out, (hue[0], sat[0], val[0]),  (hue[1], sat[1], val[1]))

    @staticmethod
    def __find_contours(input, external_only):
        """Sets the values of pixels in a binary image to their distance to the nearest black pixel.
        Args:
            input: A numpy.ndarray.
            external_only: A boolean. If true only external contours are found.
        Return:
            A list of numpy.ndarray where each one represents a contour.
        """
        if(external_only):
            mode = cv2.RETR_EXTERNAL
        else:
            mode = cv2.RETR_LIST
        method = cv2.CHAIN_APPROX_SIMPLE
        #im2, contours, hierarchy =cv2.findContours(input, mode=mode, method=method)
        contours, hierarchy =cv2.findContours(input, mode=mode, method=method)
        return contours


    @staticmethod
    def calcCX(output):
        def calc_center(M):
            """Detect the center given the moment of a contour."""
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            return cx, cy
        def polygon(c):
            """Remove concavities from a contour and turn it into a polygon."""
            hull = cv2.convexHull(c)
            epsilon = 0.025 * cv2.arcLength(hull, True)
            goal = cv2.approxPolyDP(hull, epsilon, True)
            return goal
        c = max(output, key=cv2.contourArea)
        # make sure the largest contour is significant
        area = cv2.contourArea(c)
        if area > 0:
                # make suggested contour into a polygon
                goal = polygon(c)
        M = cv2.moments(c)#goal
        if M['m00'] > 0:
            cx, cy = calc_center(M)
            center = (cx, cy)
            print("CX=" + str(cx))
        return str(cx)

    @staticmethod
    def __filter_contours(input_contours, min_area, min_perimeter, min_width, max_width,
                        min_height, max_height, solidity, max_vertex_count, min_vertex_count,
                        min_ratio, max_ratio):
        """Filters out contours that do not meet certain criteria.
        Args:
            input_contours: Contours as a list of numpy.ndarray.
            min_area: The minimum area of a contour that will be kept.
            min_perimeter: The minimum perimeter of a contour that will be kept.
            min_width: Minimum width of a contour.
            max_width: MaxWidth maximum width.
            min_height: Minimum height.
            max_height: Maximimum height.
            solidity: The minimum and maximum solidity of a contour.
            min_vertex_count: Minimum vertex Count of the contours.
            max_vertex_count: Maximum vertex Count.
            min_ratio: Minimum ratio of width to height.
            max_ratio: Maximum ratio of width to height.
        Returns:
            Contours as a list of numpy.ndarray.
        """
        output = []
        for contour in input_contours:
            x,y,w,h = cv2.boundingRect(contour)
            if (w < min_width or w > max_width):
                continue
            if (h < min_height or h > max_height):
                continue
            area = cv2.contourArea(contour)
            if (area < min_area):
                continue
            if (cv2.arcLength(contour, True) < min_perimeter):
                continue
            hull = cv2.convexHull(contour)
            solid = 100 * area / cv2.contourArea(hull)
            if (solid < solidity[0] or solid > solidity[1]):
                continue
            if (len(contour) < min_vertex_count or len(contour) > max_vertex_count):
                continue
            ratio = (float)(w) / h
            if (ratio < min_ratio or ratio > max_ratio):
                continue
            output.append(contour)


        return output



if __name__ == '__main__':
    main()
