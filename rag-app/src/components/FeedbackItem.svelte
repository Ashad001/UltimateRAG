<script>
    import Card from "./card.svelte";
    import {createEventDispatcher} from 'svelte';

    export let item = {id: 0, rating: 0, text: ''};
    
    const dispatch = createEventDispatcher();

    const handleDelete = (itemId) => {
        dispatch('delete-feedback', {itemId});
    }

</script>

<Card>
<div class="num-display">
    <p>{item.rating}</p>
</div>
<button class="close" on:click={() => handleDelete(item.id)}>x</button>
<div class="feedback">
    <p>{item.text}</p>
</div>
</Card>

<style>
 .num-display {
    position: absolute;
    top: -10px;
    left: -10px;
    width: 50px;
    height: 50px;
    background: #ff6e95;
    color: #fff;
    border: 1px #eee solid;
    border-radius: 50%;
    padding: 10px;
    text-align: center;
    font-size: 19px;
 }

 .close {
    position: absolute;
    top: 10px;
    right: 20px;
    cursor: pointer;
}

</style>