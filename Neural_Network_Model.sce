
clc; clear;

// PARAMETER 
CSV_PATH    = "C:\Users\Admin\Downloads\wisman_bali_bulanan_2023_2025.csv";
N_INPUT     = 3;
N_HIDDEN    = 2;
LR          = 0.5;
EPOCHS      = 30000;
train_ratio = 0.8;
N_AHEAD     = 12;      
BOBOT_TETAP = %t;      
SEED        = 1;
RETRAIN_FULL = %t;

// FUNGSI 
function y = sigmoid(z)
    y = 1 ./ (1 + exp(-z));
endfunction

function r = rmse(a, b)
    r = sqrt(mean((a - b) .^ 2));
endfunction

function r = mape(a, b)
    r = mean(abs((a - b) ./ a)) * 100;
endfunction

function lbl = label_bulan(t)
    BULAN_ID = ["Jan","Feb","Mar","Apr","Mei","Jun","Jul","Agu","Sep","Okt","Nov","Des"];
    lbl = emptystr(size(t, "*"), 1);
    for i = 1:size(t, "*")
        thn = floor(t(i) / 12);
        bln = modulo(t(i), 12) + 1;
        lbl(i) = BULAN_ID(bln) + " " + msprintf("%02d", modulo(thn, 100));
    end
endfunction

// Forward propagation
function [h, y_hat] = forward(X, W1, b1, W2, b2)
    m     = size(X, 1);
    z1    = X * W1 + ones(m, 1) * b1;
    h     = sigmoid(z1);
    y_hat = h * W2 + b2;
endfunction

// Bobot awal
function [W1, b1, W2, b2] = bobot_awal(tetap, n_in, n_hid, seed)
    if tetap then
        W1 = [ 0.1727920960,  0.4108090718;
               0.1652185381, -0.6515786158;
               0.4526779333,  0.2231872862 ];   
        W2 = [ -0.2684766177;
                0.2905590521 ];                 
    else
        grand("setsd", seed);
        W1 = grand(n_in,  n_hid, "nor", 0, 0.5);
        W2 = grand(n_hid, 1,     "nor", 0, 0.5);
    end
    b1 = zeros(1, n_hid);
    b2 = 0;
endfunction

// Training (gradient descent, batch penuh)
function [W1, b1, W2, b2, loss] = latih(X, Y, W1, b1, W2, b2, LR, EPOCHS)
    n    = size(X, 1);
    loss = zeros(EPOCHS, 1);
    for epoch = 1:EPOCHS
        // (a) forward propagation
        [h, y_hat] = forward(X, W1, b1, W2, b2);

        // (b) error: E = 1/2 * rata-rata (y_hat - y)^2
        err = y_hat - Y;
        loss(epoch) = 0.5 * mean(err .^ 2);

        // (c) backpropagation (aturan rantai)
        dW2 = h' * err / n;
        db2 = mean(err);
        dh  = err * W2';
        dz1 = dh .* h .* (1 - h);
        dW1 = X' * dz1 / n;
        db1 = mean(dz1, "r");

        // (d) update bobot
        W2 = W2 - LR * dW2;   b2 = b2 - LR * db2;
        W1 = W1 - LR * dW1;   b1 = b1 - LR * db1;

        if or(epoch == [1 10 100 1000 5000 15000 30000 EPOCHS]) then
            mprintf("Epoch %6d | Error E = %.6f\n", epoch, loss(epoch));
        end
    end
endfunction

// Peramalan rekursif: hasil prediksi dimasukkan kembali sebagai input
function pred_n = ramal(w, W1, b1, W2, b2, n_ahead)
    pred_n = zeros(n_ahead, 1);
    for i = 1:n_ahead
        pred_n(i) = sigmoid(w * W1 + b1) * W2 + b2;
        w = [w(2:$), pred_n(i)];     // buang yang tertua, tambah prediksi
    end
endfunction

//  BACA DATA 
raw = csvRead(CSV_PATH, ",", ".", "string");
raw = raw(2:$, :);
tgl_str = raw(:, 1);
Wisman  = strtod(raw(:, 2));

// Urutkan berdasarkan tanggal
[tgl_str, idx] = gsort(tgl_str, "g", "i");
Wisman = Wisman(idx);

thn      = strtod(part(tgl_str, 1:4));
bln      = strtod(part(tgl_str, 6:7));
bulan_no = thn * 12 + (bln - 1);
N        = size(Wisman, 1);

// Bentuk data: Y(t), Y(t+1), ..., Y(t+N_INPUT)
n = N - N_INPUT;
D = zeros(n, N_INPUT + 1);
for k = 0:N_INPUT
    D(:, k + 1) = Wisman(k + 1 : k + n);
end
bulan_awal = bulan_no(1:n);

// Normalisasi Min-Max
mn = min(D);
mx = max(D);
D_n = (D - mn) / (mx - mn);
disp("Data ternormalisasi (5 baris pertama):");
disp(D_n(1:5, :));

// Split train / test
n_train = floor(n * train_ratio);
n_test  = n - n_train;
mprintf("Jumlah sampel: %d  | train = %d  | test = %d\n", n, n_train, n_test);

X_train = D_n(1:n_train, 1:$-1);     y_train = D_n(1:n_train, $);
X_test  = D_n(n_train+1:$, 1:$-1);   y_test  = D_n(n_train+1:$, $);

// TRAINING (data train) 
[W1_0, b1_0, W2_0, b2_0] = bobot_awal(BOBOT_TETAP, N_INPUT, N_HIDDEN, SEED);
[W1, b1, W2, b2, loss_history] = latih(X_train, y_train, W1_0, b1_0, W2_0, b2_0, LR, EPOCHS);

// EVALUASI (skala asli) 
[tmp, pred_train_n] = forward(X_train, W1, b1, W2, b2);
[tmp, pred_test_n]  = forward(X_test,  W1, b1, W2, b2);

y_train_real = y_train      * (mx - mn) + mn;
pred_train   = pred_train_n * (mx - mn) + mn;
y_test_real  = y_test       * (mx - mn) + mn;
pred_test    = pred_test_n  * (mx - mn) + mn;

// pembanding naive: prediksi = nilai terakhir yang diketahui (kolom input terakhir)
naive_test = D(n_train+1:$, N_INPUT);

mprintf("\n=== HASIL EVALUASI (skala asli, jumlah wisatawan) ===\n");
mprintf("RMSE train : %.0f   | MAPE train : %.2f%%\n", rmse(y_train_real, pred_train), mape(y_train_real, pred_train));
mprintf("RMSE test  : %.0f   | MAPE test  : %.2f%%\n", rmse(y_test_real,  pred_test),  mape(y_test_real,  pred_test));
mprintf("RMSE naive (pembanding) test : %.0f\n", rmse(y_test_real, naive_test));

bulan_target = bulan_awal + N_INPUT;     // bulan yang diprediksi
lbl_all      = label_bulan(bulan_target);
mprintf("\n  bulan_target     aktual    prediksi\n");
for i = 1:n_test
    mprintf("  %-12s %10.0f  %10.0f\n", lbl_all(n_train + i), y_test_real(i), pred_test(i));
end

// PERAMALAN DI LUAR DATA 
if RETRAIN_FULL then
    mprintf("\n--- Latih ulang pada seluruh data untuk peramalan ---\n");
    [W1f, b1f, W2f, b2f] = latih(D_n(:, 1:$-1), D_n(:, $), W1_0, b1_0, W2_0, b2_0, LR, EPOCHS);
else
    W1f = W1;  b1f = b1;  W2f = W2;  b2f = b2;
end

w0        = D_n(n, 2:$);                 // 3 bulan terakhir data (baris 1x3)
pn        = ramal(w0, W1f, b1f, W2f, b2f, N_AHEAD);
pred_next = pn * (mx - mn) + mn;         // kembalikan ke skala asli

bulan_next = bulan_no(N) + (1:N_AHEAD);  // mulai satu bulan setelah data terakhir
lbl_next   = label_bulan(bulan_next);

mprintf("\n=== PREDIKSI (jumlah wisatawan) ===\n");
for i = 1:N_AHEAD
    mprintf("%-8s %10.0f\n", lbl_next(i), pred_next(i));
end

// =====================================================================
// GRAFIK 1: Error training per epoch
// =====================================================================
f1 = scf(1); clf(f1);
f1.figure_size = [800, 500];

plot2d(1:EPOCHS, loss_history, style = 2, logflag = "nl");
xtitle("Error training per epoch", "Epoch", "E = 1/2 MSE (ternormalisasi)");
xgrid(color("grey"));

xs2png(f1, "grafik1_error_training.png");

// =====================================================================
// GRAFIK 2: Aktual vs prediksi (train & test)
// =====================================================================
f2 = scf(2); clf(f2);
f2.figure_size = [1000, 500];

x_all = (1:n)';
plot(x_all, [y_train_real; y_test_real], "bo-");
plot(x_all(1:n_train), pred_train, "gs--");
plot(x_all(n_train+1:$), pred_test, "rs--");
legend(["Aktual", "Prediksi (train)", "Prediksi (test)"], "in_upper_left");
a2 = gca();
a2.font_size = 2;
a2.data_bounds(:, 1) = [0.5; n + 0.5];
// garis pembatas train/test
yb = a2.data_bounds(:, 2);
plot([n_train + 0.5, n_train + 0.5], yb', "k:");
// label sumbu-x
tk = 1:3:n;
a2.x_ticks = tlist(["ticks", "locations", "labels"], tk', lbl_all(tk));
xtitle("Kedatangan Wisman Bali: aktual vs prediksi", "", "Jumlah wisatawan");
xgrid(color("grey"));

xs2png(f2, "grafik2_aktual_vs_prediksi.png");

// =====================================================================
// GRAFIK 3: Aktual dan prediksi di luar data (2026)
// =====================================================================
f3 = scf(3); clf(f3);
f3.figure_size = [1000, 500];

x_aktual = (N_INPUT + 1):N;              // indeks bulan target pada data aktual
x_pred   = (N + 1):(N + N_AHEAD);        // indeks bulan peramalan
plot(x_aktual, D(:, $)', "o-b");
plot(x_pred, pred_next', "s--r");
legend(["Aktual", "Prediksi 2026"], "in_upper_left");
a3 = gca();
a3.font_size = 2;
a3.data_bounds(:, 1) = [N_INPUT + 0.5; N + N_AHEAD + 0.5];
// garis pembatas data aktual / peramalan
yb3 = a3.data_bounds(:, 2);
plot([N + 0.5, N + 0.5], yb3', "k:");
// label sumbu-x (indeks 1 = bulan pertama pada data)
tk3  = (N_INPUT + 1):3:(N + N_AHEAD);
lbl3 = label_bulan(bulan_no(1) + tk3 - 1);
a3.x_ticks = tlist(["ticks", "locations", "labels"], tk3', lbl3);
xtitle("Kedatangan Wisman Bali: aktual dan prediksi 2026", "", "Jumlah wisatawan");
xgrid(color("grey"));

xs2png(f3, "grafik3_prediksi_2026.png");

mprintf("\nSelesai. 3 file PNG tersimpan di: %s\n", pwd());
