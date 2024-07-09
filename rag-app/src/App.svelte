<script>
	import FeedbackList from "./components/FeedbackList.svelte";
	import FeedbackStats from "./components/FeedbackStats.svelte"
	import FeedbackForm from "./components/feedbackForm.svelte";
	let feedback = [
		{
			id: 1,
			rating: 5,
			text: 'Great app!'
		},
		{
			id: 2,
			rating: 3,
			text: 'Could be better'
		},
		{
			id: 3,
			rating: 4,
			text: 'Awesome!'
		}
	]
	
	$: count = feedback.length;
	$: average = feedback.reduce((acc, curr) => acc + curr.rating, 0) / count;


	const deleteFeedback = (e) => {
		feedback = feedback.filter(fb => fb.id !== e.detail.itemId);
	}

</script>

<main class="container">
	<FeedbackForm />
	<FeedbackStats {count} {average}/>
	<FeedbackList {feedback} on:delete-feedback = {deleteFeedback}/>
</main>



