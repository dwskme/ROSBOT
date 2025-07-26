import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms
import os
import time
import math


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        def CBR(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        self.enc1 = CBR(3, 64)
        self.enc2 = CBR(64, 128)
        self.enc3 = CBR(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.dec2 = CBR(256 + 128, 128)
        self.dec1 = CBR(128 + 64, 64)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.final = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = self.up(e3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return torch.sigmoid(self.final(d1))


class LaneFollowerNode(Node):
    def __init__(self):
        super().__init__('lane_follower_angle')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(Image, '/camera/color/image_raw', self.image_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.debug_pub = self.create_publisher(Image, '/lane_debug', 10)

        # Load model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UNet().to(self.device)
        model_path = os.path.join(os.path.dirname(__file__), 'lane_unet.pth')

        if not os.path.exists(model_path):
            self.get_logger().error(f"❌ Model file not found: {model_path}")
            rclpy.shutdown()
            return

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.get_logger().info("✅ Lane follower with Hough angle steering initialized.")

    def filter_mask(self, mask):
        h, w = mask.shape
        mask[:int(h * 0.5), :] = 0
        kernel = np.ones((5, 5), np.uint8)
        return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)

    def detect_hough_lines(self, mask):
        blurred = cv2.GaussianBlur(mask * 255, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        return cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=20, maxLineGap=15)

    def compute_average_angle(self, lines):
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = math.atan2(y2 - y1, x2 - x1)
            angles.append(angle)
        return np.mean(angles) if angles else None

    def draw_hough_lines(self, image, lines):
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        return image

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Image error: {e}")
            return

        resized = cv2.resize(frame, (256, 256))
        tensor = self.transform(resized).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred = self.model(tensor).squeeze().cpu().numpy()

        blurred_pred = cv2.GaussianBlur(pred, (5, 5), 0)
        raw_mask = (blurred_pred > 0.4).astype(np.uint8)
        mask = self.filter_mask(raw_mask)

        lines = self.detect_hough_lines(mask)

        if lines is not None:
            avg_angle = self.compute_average_angle(lines)
            degrees_angle = math.degrees(avg_angle)
            angular_z = -0.8 * avg_angle  # Gain-tuned
            linear_x = 0.15
            self.get_logger().info(f"📐 Hough angle = {degrees_angle:.2f}°, angular_z = {angular_z:.3f}")
        else:
            angular_z = 0.0
            linear_x = 0.0
            self.get_logger().warn("⚠️ No lines detected.")

        self.publish_cmd(linear_x, angular_z)

        overlay = cv2.addWeighted(resized, 0.7, cv2.applyColorMap(mask * 255, cv2.COLORMAP_JET), 0.3, 0)
        overlay = self.draw_hough_lines(overlay, lines)
        debug_msg = self.bridge.cv2_to_imgmsg(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), encoding="rgb8")
        self.debug_pub.publish(debug_msg)

    def publish_cmd(self, lin, ang):
        twist = Twist()
        twist.linear.x = lin
        twist.angular.z = ang
        self.cmd_pub.publish(twist)

    def stop_bot(self):
        self.publish_cmd(0.0, 0.0)
        self.get_logger().info("🛑 Bot stopped.")


def main(args=None):
    rclpy.init(args=args)
    node = LaneFollowerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("🔻 Ctrl+C")
    finally:
        node.stop_bot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
