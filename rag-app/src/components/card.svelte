<div class="card">
    <slot>
        <div class="card">
            <div class="card-header">
                <slot name="header">
                    <h2>Card Header</h2>
                </slot>
            </div>
            <div class="card-body">
                <slot name="body">
                    <p>Card Body</p>
                </slot>
            </div>
            <div class="card-footer">
                <slot name="footer">
                    <p>Card Footer</p>
                </slot>
            </div>
        </div>
    </slot>
</div>

<style>
    .card {
        border: 1px solid #ccc;
        border-radius: 15px;
        color: #333;
        margin: 20px 0px;
        position: relative;
        padding: 40px 50px;
        background-color: white;
    }

    .card-header {
        background-color: #f0f0f0;
        border-bottom: 1px solid #ccc;
        border-top-left-radius: 5px;
        border-top-right-radius: 5px;
        padding: 10px;
    }

    .card-body {
        padding: 10px;
    }

    .card-footer {
        background-color: #f0f0f0;
        border-bottom-left-radius: 5px;
        border-bottom-right-radius: 5px;
        border-top: 1px solid #ccc;
        padding: 10px;
    }
</style>