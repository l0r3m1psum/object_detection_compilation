// capture_frame_png.c

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/videodev2.h>
#include <sys/mman.h>

#define DEVICE "/dev/video0"
#define WIDTH 640
#define HEIGHT 320

int main() {
    int fd = open(DEVICE, O_RDWR);
    if (fd == -1) {
        perror("Opening video device");
        return 1;
    }

    // Set format
    struct v4l2_format format;
    memset(&format, 0, sizeof(format));
    format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    format.fmt.pix.width = WIDTH;
    format.fmt.pix.height = HEIGHT;
    format.fmt.pix.pixelformat = V4L2_PIX_FMT_RGB24;
    format.fmt.pix.field = V4L2_FIELD_NONE;

    if (ioctl(fd, VIDIOC_S_FMT, &format) == -1) {
        perror("Setting Pixel Format");
        close(fd);
        return 1;
    }

    // Request buffer
    struct v4l2_requestbuffers reqbuf;
    memset(&reqbuf, 0, sizeof(reqbuf));
    reqbuf.count = 1;
    reqbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    reqbuf.memory = V4L2_MEMORY_MMAP;

    if (ioctl(fd, VIDIOC_REQBUFS, &reqbuf) == -1) {
        perror("Requesting Buffer");
        close(fd);
        return 1;
    }

    // Query the buffer
    struct v4l2_buffer buffer;
    memset(&buffer, 0, sizeof(buffer));
    buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buffer.memory = V4L2_MEMORY_MMAP;
    buffer.index = 0;

    if (ioctl(fd, VIDIOC_QUERYBUF, &buffer) == -1) {
        perror("Querying Buffer");
        close(fd);
        return 1;
    }

    void *buffer_start = mmap(NULL, buffer.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, buffer.m.offset);
    if (buffer_start == MAP_FAILED) {
        perror("mmap");
        close(fd);
        return 1;
    }

    // Queue the buffer
    if (ioctl(fd, VIDIOC_QBUF, &buffer) == -1) {
        perror("Queue Buffer");
        close(fd);
        return 1;
    }

    // Start streaming
    int type = buffer.type;
    if (ioctl(fd, VIDIOC_STREAMON, &type) == -1) {
        perror("Start Capture");
        close(fd);
        return 1;
    }

    // Dequeue buffer to get the frame
    if (ioctl(fd, VIDIOC_DQBUF, &buffer) == -1) {
        perror("Retrieving Frame");
        close(fd);
        return 1;
    }

    // Save as PNG using stb_image_write
#define RGB 1
    int quality = 100;
    if (!stbi_write_jpg("frame.jpg", WIDTH, HEIGHT, RGB, buffer.start, quality)) {
        fprintf(stderr, "Failed to write image\n");
    } else {
        printf("Saved frame to frame.jpg\n");
    }

    // Cleanup
    munmap(buffer_start, buffer.length);
    ioctl(fd, VIDIOC_STREAMOFF, &type);
    close(fd);

    return 0;
}