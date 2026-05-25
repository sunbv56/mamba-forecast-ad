import os
import matplotlib
matplotlib.use('Agg')  # Sử dụng Agg backend để chạy không cần GUI
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_architecture_diagram():
    # Khởi tạo khung hình
    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis('off')
    
    # Bảng màu sắc học thuật (Soft Pastels)
    colors = {
        'data': '#F3E8FF',      # Tím nhạt
        'stats': '#FFE6E6',     # Đỏ nhạt (vật lý)
        'decomp': '#E0F2FE',    # Xanh dương nhạt (phân tách)
        'mamba': '#DCFCE7',     # Xanh lá nhạt (Mamba)
        'fusion': '#FFEDD5',    # Cam nhạt (dung hợp)
        'output': '#FEF9C3',    # Vàng nhạt (đầu ra)
        'border': '#4B5563',    # Xám viền
        'text': '#1F2937'       # Đen xám text
    }

    # Tiêu đề hình vẽ
    ax.text(8, 8.5, "Proposed Physics-Informed Channel-Independent Hybrid Mamba++ Anomaly Detection Framework",
            fontsize=15, fontweight='bold', ha='center', color='#111827')

    # --- 1. TIỀN XỬ LÝ DỮ LIỆU THÔ ---
    rect_data = patches.FancyBboxPatch((0.5, 3.5), 2.2, 3.5, boxstyle="round,pad=0.1", 
                                       facecolor=colors['data'], edgecolor=colors['border'], linewidth=1.5)
    ax.add_patch(rect_data)
    ax.text(1.6, 6.7, "1. DATA ACQUISITION & SLICING", fontsize=9, fontweight='bold', ha='center', color=colors['text'])
    ax.text(1.6, 5.8, r"Raw Acceleration" + "\n" + r"(Horizontal & Vertical)" + "\n" + r"$x \in \mathbb{R}^{2 \times L_{total}}$", 
            fontsize=8, ha='center', color=colors['text'])
    # Hộp Lookback Window
    rect_win = patches.FancyBboxPatch((0.7, 3.8), 1.8, 1.2, boxstyle="round,pad=0.05",
                                      facecolor='#FFFFFF', edgecolor='#9CA3AF', linewidth=1)
    ax.add_patch(rect_win)
    ax.text(1.6, 4.4, r"Sliding Lookback Window" + "\n" + r"$x_{raw} \in \mathbb{R}^{2 \times L}$ (L=1024)", 
            fontsize=7.5, ha='center', color=colors['text'], fontweight='semibold')

    # --- 2. NHÁNH TRÍCH XUẤT ĐẶC TRƯNG VẬT LÝ ---
    rect_stats = patches.FancyBboxPatch((3.4, 0.8), 2.4, 2.5, boxstyle="round,pad=0.1", 
                                        facecolor=colors['stats'], edgecolor=colors['border'], linewidth=1.5)
    ax.add_patch(rect_stats)
    ax.text(4.6, 3.0, "2. PHYSICAL STATS BRANCH", fontsize=9, fontweight='bold', ha='center', color=colors['text'])
    ax.text(4.6, 2.2, r"8 Time-Domain Features" + "\n" + r"(RMS, Kurtosis, Skewness," + "\n" + r"Peak-to-Peak, Mean, Std," + "\n" + r"Crest & Shape Factors)" + "\n" + r"$stats \in \mathbb{R}^{2 \times 8}$", 
            fontsize=8, ha='center', color=colors['text'])
    
    # Hộp BatchNorm1d trong Model
    rect_bn = patches.FancyBboxPatch((3.6, 1.0), 2.0, 0.5, boxstyle="round,pad=0.05",
                                     facecolor='#FFFFFF', edgecolor='#EF4444', linewidth=1.2)
    ax.add_patch(rect_bn)
    ax.text(4.6, 1.2, "Model BatchNorm1d\n(Avoid dominance)", fontsize=7.5, ha='center', color='#B91C1C', fontweight='bold')

    # --- 3. PHÂN RÃ CHUỖI THỜI GIAN ---
    rect_decomp = patches.FancyBboxPatch((3.4, 4.5), 2.4, 2.5, boxstyle="round,pad=0.1", 
                                         facecolor=colors['decomp'], edgecolor=colors['border'], linewidth=1.5)
    ax.add_patch(rect_decomp)
    ax.text(4.6, 6.7, "3. SERIES DECOMPOSITION", fontsize=9, fontweight='bold', ha='center', color=colors['text'])
    
    # Nhánh Seasonal
    rect_seas = patches.FancyBboxPatch((3.6, 5.7), 2.0, 0.6, boxstyle="round,pad=0.05",
                                       facecolor='#FFFFFF', edgecolor='#0284C7', linewidth=1)
    ax.add_patch(rect_seas)
    ax.text(4.6, 5.9, r"Seasonal Component" + "\n" + r"$x_{seas} \in \mathbb{R}^{2 \times L}$", fontsize=7.5, ha='center', color='#0369A1', fontweight='semibold')
    
    # Nhánh Trend
    rect_trend = patches.FancyBboxPatch((3.6, 4.7), 2.0, 0.6, boxstyle="round,pad=0.05",
                                        facecolor='#FFFFFF', edgecolor='#0284C7', linewidth=1)
    ax.add_patch(rect_trend)
    ax.text(4.6, 4.9, r"Trend Component" + "\n" + r"$x_{trend} \in \mathbb{R}^{2 \times L}$", fontsize=7.5, ha='center', color='#0369A1', fontweight='semibold')

    # --- 4. CI-MAMBA ENCODER & POOLING ---
    rect_mamba = patches.FancyBboxPatch((6.5, 4.0), 2.6, 3.0, boxstyle="round,pad=0.1", 
                                        facecolor=colors['mamba'], edgecolor=colors['border'], linewidth=1.5)
    ax.add_patch(rect_mamba)
    ax.text(7.8, 6.7, "4. CI-MAMBA ENCODER", fontsize=9, fontweight='bold', ha='center', color=colors['text'])
    
    # Step 1: Patching & Fold
    rect_patch = patches.FancyBboxPatch((6.7, 5.7), 2.2, 0.6, boxstyle="round,pad=0.05",
                                        facecolor='#FFFFFF', edgecolor='#15803D', linewidth=1)
    ax.add_patch(rect_patch)
    ax.text(7.8, 5.9, r"Simple Patching & Folding" + "\n" + r"$x_{seas} \to (B \cdot C, N, d)$", fontsize=7.5, ha='center', color='#166534')
    
    # Step 2: Mamba
    rect_mambablock = patches.FancyBboxPatch((6.7, 4.9), 2.2, 0.6, boxstyle="round,pad=0.05",
                                             facecolor='#FFFFFF', edgecolor='#15803D', linewidth=1.2)
    ax.add_patch(rect_mambablock)
    ax.text(7.8, 5.1, "Mamba Selective Scan\n(Captures dynamics)", fontsize=7.5, ha='center', color='#166534', fontweight='bold')
    
    # Step 3: Pooling
    rect_pool = patches.FancyBboxPatch((6.7, 4.1), 2.2, 0.6, boxstyle="round,pad=0.05",
                                       facecolor='#FFFFFF', edgecolor='#15803D', linewidth=1)
    ax.add_patch(rect_pool)
    ax.text(7.8, 4.3, r"Avg + Max Pooling" + "\n" + r"$s_{flat} \in \mathbb{R}^{B \cdot C \times d}$", fontsize=7.5, ha='center', color='#166534')

    # --- 5. FUSION HEAD ---
    rect_fusion = patches.FancyBboxPatch((9.8, 2.0), 2.4, 4.0, boxstyle="round,pad=0.1", 
                                         facecolor=colors['fusion'], edgecolor=colors['border'], linewidth=1.5)
    ax.add_patch(rect_fusion)
    ax.text(11.0, 5.7, "5. FUSION & PROJECTION", fontsize=9, fontweight='bold', ha='center', color=colors['text'])
    
    # Hộp Concat
    rect_concat = patches.FancyBboxPatch((10.0, 4.3), 2.0, 0.6, boxstyle="round,pad=0.05",
                                         facecolor='#FFFFFF', edgecolor='#EA580C', linewidth=1.2)
    ax.add_patch(rect_concat)
    ax.text(11.0, 4.5, "Concatenate Feature Map\n$[s_{flat}, stats_{norm}]$", fontsize=7.5, ha='center', color='#C2410C', fontweight='bold')
    
    # Hộp Linear Projection
    rect_proj = patches.FancyBboxPatch((10.0, 3.3), 2.0, 0.6, boxstyle="round,pad=0.05",
                                        facecolor='#FFFFFF', edgecolor='#EA580C', linewidth=1)
    ax.add_patch(rect_proj)
    ax.text(11.0, 3.5, r"Linear Projection" + "\n" + r"Forecast Seasonal $\to H$", fontsize=7.5, ha='center', color='#C2410C')
    
    # Hộp Unfold
    rect_unfold = patches.FancyBboxPatch((10.0, 2.3), 2.0, 0.6, boxstyle="round,pad=0.05",
                                         facecolor='#FFFFFF', edgecolor='#EA580C', linewidth=1)
    ax.add_patch(rect_unfold)
    ax.text(11.0, 2.5, r"CI Channel Unfolding" + "\n" + r"$y_{seasonal} \in \mathbb{R}^{B \times C \times H}$", fontsize=7.5, ha='center', color='#C2410C')

    # --- 6. OUTPUT & AD DECISION ---
    rect_out = patches.FancyBboxPatch((12.9, 1.2), 2.6, 5.0, boxstyle="round,pad=0.1", 
                                      facecolor=colors['output'], edgecolor=colors['border'], linewidth=1.5)
    ax.add_patch(rect_out)
    ax.text(14.2, 5.9, "6. AD DECISION HEAD", fontsize=9, fontweight='bold', ha='center', color=colors['text'])
    
    # Mixing Layer
    rect_mix = patches.FancyBboxPatch((13.1, 4.8), 2.2, 0.7, boxstyle="round,pad=0.05",
                                      facecolor='#FFFFFF', edgecolor='#CA8A04', linewidth=1.2)
    ax.add_patch(rect_mix)
    ax.text(14.2, 5.0, r"Learnable Mixing Layer" + "\n" + r"$y_{pred} = \alpha y_{seas} + (1-\alpha)y_{trend}$", 
            fontsize=7.5, ha='center', color='#854D0E', fontweight='bold')
    
    # MSE Scorer
    rect_mse = patches.FancyBboxPatch((13.1, 3.4), 2.2, 0.6, boxstyle="round,pad=0.05",
                                      facecolor='#FFFFFF', edgecolor='#CA8A04', linewidth=1)
    ax.add_patch(rect_mse)
    ax.text(14.2, 3.6, r"MSE Anomaly Score" + "\n" + r"$Score = \|y - y_{pred}\|^2$", fontsize=7.5, ha='center', color='#854D0E')
    
    # POT Decision
    rect_pot = patches.FancyBboxPatch((13.1, 1.5), 2.2, 1.1, boxstyle="round,pad=0.05",
                                      facecolor='#FEF2F2', edgecolor='#EF4444', linewidth=1.5)
    ax.add_patch(rect_pot)
    ax.text(14.2, 2.3, "Dynamic Thresholding", fontsize=8, ha='center', color='#B91C1C', fontweight='bold')
    ax.text(14.2, 1.7, "Peak-Over-Threshold\n(POT) Decision\nHealthy (0) / Alarm (1)", fontsize=7.5, ha='center', color='#B91C1C')

    # --- ĐƯỜNG MŨI TÊN KẾT NỐI (ARROWS) ---
    arrow = dict(facecolor='#4B5563', edgecolor='#4B5563', width=1.5, headwidth=6, shrink=0.08)
    
    # Data -> Decomp & Stats
    ax.annotate("", xy=(3.3, 5.8), xytext=(2.8, 5.3), arrowprops=arrow)
    ax.annotate("", xy=(3.3, 2.5), xytext=(2.8, 5.3), arrowprops=arrow)
    
    # Decomp (Seasonal) -> Mamba
    ax.annotate("", xy=(6.4, 6.0), xytext=(5.7, 6.0), arrowprops=arrow)
    
    # Decomp (Trend) -> Mix Layer (vòng qua bên dưới hoặc ngang qua)
    ax.annotate("", xy=(13.0, 5.2), xytext=(5.7, 5.0), 
                arrowprops=dict(arrowstyle="->", color='#0284C7', lw=1.5, connectionstyle="angle,angleA=0,angleB=90,rad=10"))

    # Mamba -> Fusion (Concat)
    ax.annotate("", xy=(9.9, 4.6), xytext=(9.2, 5.5), arrowprops=arrow)
    
    # Stats (BatchNorm) -> Fusion (Concat)
    ax.annotate("", xy=(9.9, 4.4), xytext=(5.7, 1.25), 
                arrowprops=dict(arrowstyle="->", color='#EF4444', lw=1.5, connectionstyle="angle,angleA=0,angleB=90,rad=10"))

    # Fusion (Unfold) -> Mixing Layer
    ax.annotate("", xy=(13.0, 5.3), xytext=(12.1, 2.6), 
                arrowprops=dict(arrowstyle="->", color='#EA580C', lw=1.5, connectionstyle="angle,angleA=0,angleB=90,rad=10"))

    # Mixing -> MSE -> POT
    ax.annotate("", xy=(14.2, 4.1), xytext=(14.2, 4.7), arrowprops=arrow)
    ax.annotate("", xy=(14.2, 2.7), xytext=(14.2, 3.3), arrowprops=arrow)

    # --- CHÚ THÍCH CÁC KÊNH LUỒNG (LEGEND) ---
    rect_legend = patches.FancyBboxPatch((0.5, 0.4), 15.0, 0.5, boxstyle="round,pad=0.05",
                                         facecolor='#F3F4F6', edgecolor='#D1D5DB', linewidth=1)
    ax.add_patch(rect_legend)
    
    # Các nhãn chú thích nằm ngang
    ax.text(1.0, 0.6, "LEGEND:", fontsize=8.5, fontweight='bold', color='#374151')
    
    ax.plot([2.5, 3.2], [0.65, 0.65], color='#4B5563', lw=2)
    ax.text(3.4, 0.6, "Standard Flow", fontsize=8, color='#374151')
    
    ax.plot([5.5, 6.2], [0.65, 0.65], color='#0284C7', lw=2)
    ax.text(6.4, 0.6, "Trend Feature Path", fontsize=8, color='#374151')
    
    ax.plot([8.5, 9.2], [0.65, 0.65], color='#EF4444', lw=2)
    ax.text(9.4, 0.6, "Physical Stats Path", fontsize=8, color='#374151')
    
    ax.plot([11.8, 12.5], [0.65, 0.65], color='#EA580C', lw=2)
    ax.text(12.7, 0.6, "Seasonal Prediction Path", fontsize=8, color='#374151')

    # Lưu hình ảnh
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'results')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'architecture_diagram.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[SUCCESS] Diagram saved at: {output_path}")

if __name__ == "__main__":
    create_architecture_diagram()
