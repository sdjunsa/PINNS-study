import torch
import torch.nn as nn
import torch.autograd as autograd # 자동 미분을 위해 필요
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import math

# 1. 신경망
class PhysicsInformedNN(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, hidden_neurons):
        super(PhysicsInformedNN, self).__init__()

        # 신경망 층 정의
        layers = [nn.Linear(input_dim, hidden_neurons), nn.Tanh()] # 입력 층

        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_neurons, hidden_neurons), nn.Tanh()]) # 은닉 층

        layers.append(nn.Linear(hidden_neurons, output_dim)) # 출력 층

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# 2. PINNs 모델 클래스
class PINNs:
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3) # Adam 옵티마이저 사용

        # 경계 조건 지점 및 값 (PyTorch Tensor로 정의)
        self.x_bc = torch.tensor([[0.0], [math.pi/2.0]], dtype=torch.float32)
        self.y_bc = torch.tensor([[0.0], [1.0]], dtype=torch.float32)

        # PDE 손실 계산을 위한 지점 샘플링 (나중에 학습 루프에서 생성)
        self.x_pde = None

    # PDE 잔차(residual) 계산
    def pde_residual(self, x):
        # x에 대해 미분 가능하도록 설정
        x.requires_grad_(True)

        # 신경망 출력 (y_hat = y(x) 근사)
        y_hat = self.model(x)

        # y_hat에 대한 x의 1차 미분 (dy/dx)
        dy_dx = autograd.grad(y_hat, x,
                              grad_outputs=torch.ones_like(y_hat), # y_hat이 스칼라가 아닐 경우 필요
                              create_graph=True)[0] # 2차 미분을 위해 그래프 생성 유지

        # dy_dx에 대한 x의 2차 미분 (d^2y/dx^2)
        d2y_dx2 = autograd.grad(dy_dx, x,
                                grad_outputs=torch.ones_like(dy_dx),
                                create_graph=True)[0]

        # PDE 잔차: d^2y/dx^2 + y
        pde_r = d2y_dx2 + y_hat

        return pde_r

    # 총 손실 함수
    def total_loss(self, x_pde):
        # PDE 손실 (MSE)
        pde_r = self.pde_residual(x_pde)
        loss_pde = torch.mean(pde_r**2)

        # 경계 조건 손실 (MSE)
        y_bc_pred = self.model(self.x_bc)
        loss_bc = torch.mean((y_bc_pred - self.y_bc)**2)

        # 총 손실: PDE 손실 + BC 손실
        loss = loss_pde + loss_bc # 간단하게 합산 (가중치 조절 가능)

        return loss, loss_pde, loss_bc

    # 학습 함수
    def train(self, num_epochs, num_pde_points):
        history = []
        for epoch in range(num_epochs):
            self.optimizer.zero_grad() # 그래디언트 초기화

            # PDE 손실 계산을 위한 지점 샘플링 (학습 루프 내에서 매번 다르게 샘플링 가능)
            # 여기서는 간단히 고정된 linspace 사용
            self.x_pde = torch.linspace(0, math.pi/2.0, num_pde_points, requires_grad=True).reshape(-1, 1)

            # 총 손실 계산
            loss, loss_pde, loss_bc = self.total_loss(self.x_pde)

            # 역전파 및 가중치 업데이트
            loss.backward()
            self.optimizer.step()

            if (epoch + 1) % 1000 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Total Loss: {loss.item():.6f}, PDE Loss: {loss_pde.item():.6f}, BC Loss: {loss_bc.item():.6f}')
                history.append(loss.item())
        return history

    # 예측 함수
    def predict(self, x_test):
        self.model.eval() # 평가 모드
        with torch.no_grad(): # 그래디언트 계산 비활성화
            x_test_tensor = torch.tensor(x_test, dtype=torch.float32).reshape(-1, 1)
            y_pred = self.model(x_test_tensor)
        return y_pred.numpy()

# 3. 메인 실행 블록
if __name__ == "__main__":
    # 하이퍼파라미터
    input_dim = 1
    output_dim = 1
    num_layers = 4       # 은닉 층 개수
    hidden_neurons = 30  # 각 은닉 층의 뉴런 수
    num_epochs = 10000    # 학습 에포크 수
    num_pde_points = 100 # PDE 손실 계산에 사용할 지점 수

    # 모델 및 PINNs 클래스 인스턴스 생성
    model = PhysicsInformedNN(input_dim, output_dim, num_layers, hidden_neurons)
    pinn = PINNs(model)

    print("--- PINNs 모델 학습 시작 ---")
    loss_history = pinn.train(num_epochs, num_pde_points)
    print("--- 학습 완료 ---")

    # 결과 시각화
    x_test = np.linspace(0, math.pi/2.0, 100).reshape(-1, 1)
    y_pred = pinn.predict(x_test)

    # 해석해 (sin(x))
    y_analytical = np.sin(x_test)

    plt.figure(figsize=(10, 6))
    plt.plot(x_test, y_analytical, label='Analytical Solution ($y = \sin(x)$)', color='red', linewidth=2)
    plt.plot(x_test, y_pred, label='PINNs Solution', color='blue', linestyle='--')
    plt.scatter(pinn.x_bc.numpy(), pinn.y_bc.numpy(), color='green', marker='o', label='Boundary Points', s=50)
    plt.title('PINNs Solution vs Analytical Solution')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 손실 변화 그래프
    # plt.figure(figsize=(10, 6))
    # plt.plot(range(0, num_epochs, 1000), loss_history)
    # plt.title('Total Loss during Training (every 1000 epochs)')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.grid(True)
    # plt.show()
