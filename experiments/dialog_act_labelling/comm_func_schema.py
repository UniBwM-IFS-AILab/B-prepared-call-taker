from __future__ import annotations

import logging
from enum import StrEnum
from typing import Annotated, ClassVar, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

from ems_prepared.dialogue_state.generate_strict_model import inline_json_schema_refs
from ems_prepared.dialogue_state.emergency_call_state import EmergencyCall
from ems_prepared.dialogue_state.schema_variants import slot_name_type_from_model

from .enums import CommunicativeFunction, CustomFunction

LOGGER = logging.getLogger("dialog_act_labelling")
MAX_ACTS_PER_TURN = 3


class MetaFunction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    annotation_relevant: ClassVar[bool] = False
    policy_relevant: ClassVar[bool] = False


class SlotlessFunction(MetaFunction):
    model_config = ConfigDict(extra="forbid")


class SlotOnlyFunction(SlotlessFunction):
    model_config = ConfigDict(extra="forbid", frozen=True)
    slot: slot_name_type_from_model(EmergencyCall) | None


class SlotValueFunction(SlotOnlyFunction):
    value: Annotated[str, Field(min_length=1)] | bool | int | None


class Question(SlotOnlyFunction):
    annotation_relevant = True

    communicative_function: Literal[CommunicativeFunction.QUESTION] = Field(
        title="question",
        description="Communicative function of a dialogue act performed by the sender, S, in order to obtain the information, described by the semantic content, which S assumes that the addressee, A, possesses; S puts pressure on A to provide this information.",
        examples=["And so?"],
    )


class PropositionalQuestion(SlotOnlyFunction):
    annotation_relevant = False

    communicative_function: Literal[CommunicativeFunction.PROPOSITIONAL_QUESTION] = (
        Field(
            title="propositionalQuestion",
            description="Communicative function of a dialogue act performed by the sender, S, in order to know whether the proposition, described by the semantic content, is true. S assumes that the addressee, A, knows whether the proposition is true, and puts pressure on A to provide this information.",
            examples=["Does the meeting start at  ten?"],
        )
    )


class CheckQuestion(SlotOnlyFunction):
    annotation_relevant = False

    communicative_function: Literal[CommunicativeFunction.CHECK_QUESTION] = Field(
        title="checkQuestion",
        description="Communicative function of a dialogue act performed by the sender, S, in order to know whether a proposition, which forms the semantic content, is true. S holds the uncertain belief that it is true. S assumes that A knows whether the proposition is true or not, and puts pressure on A to provide this information.",
        examples=["The meeting starts at ten, right?"],
    )


class SetQuestion(SlotOnlyFunction):
    annotation_relevant = False

    communicative_function: Literal[CommunicativeFunction.SET_QUESTION] = Field(
        title="setQuestion",
        description="Communicative function of a dialogue act performed by the sender, S, in order to know which elements of a given set have a certain property specified by the semantic content.  S puts pressure on the addressee, A, to provide this information, which S assumes that A possesses. S believes that at least one element of the set has that property.",
        examples=["What time does the meeting start?", "How far is it to the station?"],
    )


class ChoiceQuestion(SlotOnlyFunction):
    annotation_relevant = False

    communicative_function: Literal[CommunicativeFunction.CHOICE_QUESTION] = Field(
        title="choiceQuestion",
        description="Communicative function of a dialogue act performed by the sender, S, in order to know which one from a list of alternative propositions, specified by the semantic content, is true; S believes that exactly one element of that list is true; S assumes that the addressee, A, knows which of the alternative propositions is true, and S puts pressure on A to provide this information.",
        examples=[
            "Should the telephone cable go in the telephone line slot or in the external line slot?"
        ],
    )


class TestQuestion(SlotlessFunction):
    annotation_relevant = False

    communicative_function: Literal[CommunicativeFunction.TEST_QUESTION] = Field(
        title="testQuestion",
        description="Communicative function of a dialogue act performed by the sender, S, in order to know whether the addressee, A, possesses the requested information, which S does possess. S puts pressure on A to provide the requested information.",
    )


class Inform(SlotlessFunction):
    annotation_relevant = True

    communicative_function: Literal[CommunicativeFunction.INFORM] = Field(
        title="inform",
        description="Information-providing act: the sender makes information available to the addressee and presents it as correct. Use inform when the speaker provides information that was not specifically asked for, even if it is medically relevant.",
        examples=["The 6.34 to Breda leaves from platform 2."],
    )


class Agreement(SlotlessFunction):
    annotation_relevant = False

    communicative_function: Literal[CommunicativeFunction.AGREEMENT] = Field(
        title="agreement",
        description="Act expressing that the sender agrees with or accepts prior information as true.",
        examples=["Exactly", "Precies!", "Netop!"],
    )


class Disagreement(SlotlessFunction):
    annotation_relevant = False

    communicative_function: Literal[CommunicativeFunction.DISAGREEMENT] = Field(
        title="disagreement",
        description="Act expressing that the sender rejects prior information as false.",
        examples=["uh… no"],
    )


class Correction(SlotlessFunction):
    annotation_relevant = True

    communicative_function: Literal[CommunicativeFunction.CORRECTION] = Field(
        title="correction",
        description="Communicative function of a dialogue act performed by the sender, S, in order to inform the addressee, A, that certain information which S has reason to believe that A assumes to be correct, is in fact incorrect and that instead the information that S provides is correct.",
        examples=["To Montreal, not to Ottawa"],
    )


class Answer(SlotOnlyFunction):
    annotation_relevant = True

    communicative_function: Literal[CommunicativeFunction.ANSWER] = Field(
        title="answer",
        description="Communicative function of a dialogue act performed by the sender, S, in order to make certain information available to the addressee, A, which S believes A wants to know; S assumes that this information is correct.",
        examples=["S: what does the display say?,  A: send error document ready"],
    )


class Confirm(SlotlessFunction):
    annotation_relevant = True

    communicative_function: Literal[CommunicativeFunction.CONFIRM] = Field(
        title="confirm",
        description="Communicative function of a dialogue act performed by the sender, S, in order to in form the addressee, A, that the proposition which forms the semantic content is true. S believes that A holds a weak belief that this proposit",
        examples=["Indeed"],
    )


class Disconfirm(SlotlessFunction):
    annotation_relevant = True

    communicative_function: Literal[CommunicativeFunction.DISCONFIRM] = Field(
        title="disconfirm",
        description="Communicative function of a dialogue act performed by the sender, S, in order to in form the addressee, A, that the proposition which forms the semantic content is false. S believes that A holds a weak belief that this proposition is true, and that S wants to know for certain whether it is; S assumes that it is false.",
        examples=["si", "jo", "toch niet", "toch wel", "doch"],
    )


class Offer(SlotlessFunction):
    annotation_relevant = False

    communicative_function: Literal[CommunicativeFunction.OFFER] = Field(
        title="offer",
        description="Commissive act where the sender conditionally commits to perform an action if the addressee consents.",
        examples=["I will look that up for you"],
    )


class Promise(SlotlessFunction):
    annotation_relevant = False

    communicative_function: Literal[CommunicativeFunction.PROMISE] = Field(
        title="promise",
        description="Commissive act where the sender commits to perform an action believed to benefit the addressee.",
        examples=["Shall I begin?", "Would you like to have some coffee?"],
    )


class AddressRequest(SlotlessFunction):
    annotation_relevant = False

    communicative_function: Literal[CommunicativeFunction.ADDRESS_REQUEST] = Field(
        title="addressRequest",
        description="Response indicating the sender is considering performing an action that was requested, possibly conditionally.",
        examples=['A: "Give me the gun." S: "If you push the bag to me"'],
    )


class AcceptRequest(SlotlessFunction):
    annotation_relevant = False

    communicative_function: Literal[CommunicativeFunction.ACCEPT_REQUEST] = Field(
        title="acceptRequest",
        description="Response committing to perform an action that was requested.",
        examples=['A: "Could you close the door please?" B: "Sure."'],
    )


class DeclineRequest(SlotlessFunction):
    annotation_relevant = False

    communicative_function: Literal[CommunicativeFunction.DECLINE_REQUEST] = Field(
        title="declineRequest",
        description="Response committing not to perform an action that was requested.",
        examples=["Not now."],
    )


class AddressSuggest(SlotlessFunction):
    annotation_relevant = False

    communicative_function: Literal[CommunicativeFunction.ADDRESS_SUGGEST] = Field(
        title="addressSuggest",
        description="Response indicating the sender is considering a suggested action, possibly under conditions.",
        examples=[
            'A: "Let\'s go together." S: "Only if we\'re in full agreement about how to proceed when we get there."'
        ],
    )


class AcceptSuggest(SlotlessFunction):
    annotation_relevant = False

    communicative_function: Literal[CommunicativeFunction.ACCEPT_SUGGEST] = Field(
        title="acceptSuggest",
        description="Response committing to perform or go along with a suggested action.",
        examples=['A: "Shall we go and have a look around?" B: "Let\'s do so."'],
    )


class DeclineSuggest(SlotlessFunction):
    annotation_relevant = False

    communicative_function: Literal[CommunicativeFunction.DECLINE_SUGGEST] = Field(
        title="declineSuggest",
        description="Response indicating the sender will not perform or go along with a suggested action.",
        examples=["I'd rather not."],
    )


class Request(SlotlessFunction):
    annotation_relevant = False

    communicative_function: Literal[CommunicativeFunction.REQUEST] = Field(
        title="request",
        description="Directive act asking the addressee to perform an action, conditional on the addressee consenting.",
        examples=["Please turn to page five", "Don't do this ever again, please"],
    )


class Instruct(SlotlessFunction):
    annotation_relevant = True

    communicative_function: Literal[CommunicativeFunction.INSTRUCT] = Field(
        title="instruct",
        description="Communicative function of a dialogue act performed by the sender, S, in order to make the addressee, A, feel obliged to perform a certain action which is described in or can be inferred from the semantic content, in the manner or with the frequency described by the semantic content. S assumes that A is able to perform this action.",
        examples=["Go right round until you get to just above that."],
    )


class Suggest(SlotlessFunction):
    annotation_relevant = False

    communicative_function: Literal[CommunicativeFunction.SUGGEST] = Field(
        title="suggest",
        description="Directive act inviting the addressee to consider an action believed to be in their interest.",
        examples=["Let's wait for the speaker to finish."],
    )


class AddressOffer(SlotlessFunction):
    annotation_relevant = False

    communicative_function: Literal[CommunicativeFunction.ADDRESS_OFFER] = Field(
        title="addressOffer",
        description="Response indicating the sender is considering whether the addressee should perform an offered action.",
        examples=["Yes please", "Je vous en prie"],
    )


class AcceptOffer(SlotlessFunction):
    annotation_relevant = False

    communicative_function: Literal[CommunicativeFunction.ACCEPT_OFFER] = Field(
        title="acceptOffer",
        description="Response indicating the sender wants the addressee to perform the offered action.",
        examples=["Yes please", "Je vous en prie", "Graag", "Bitte"],
    )


class DeclineOffer(SlotlessFunction):
    annotation_relevant = False

    communicative_function: Literal[CommunicativeFunction.DECLINE_OFFER] = Field(
        title="declineOffer",
        description="Response indicating the sender does not want the addressee to perform the offered action.",
        examples=["No thank you", "Nej tak", "Non merci"],
    )


class AutoPositive(SlotlessFunction):
    annotation_relevant = False

    communicative_function: Literal[CommunicativeFunction.AUTO_POSITIVE] = Field(
        title="autoPositive",
        description="Communicative function of a dialogue act performed by the sender, S, in order to in form the addressee, A, that S believes that S's processing of the previous utterance(s) was successful.",
        examples=["Uh-huh", "Okay", "nodding", "Yes"],
    )


class AutoNegative(SlotlessFunction):
    annotation_relevant = False

    communicative_function: Literal[CommunicativeFunction.AUTO_NEGATIVE] = Field(
        title="autoNegative",
        description="Communicative function of a dialogue act performed by the sender, S, in order to inform the addressee, A that S's processing of the previous utterance(s) encountered a problem.",
        examples=["I beg your pardon", "Como?"],
    )


class AlloPositive(SlotlessFunction):
    annotation_relevant = False

    communicative_function: Literal[CommunicativeFunction.ALLO_POSITIVE] = Field(
        title="alloPositive",
        description="Feedback act indicating the sender believes the addressee successfully processed previous utterance(s).",
        examples=["Correct", "Right"],
    )


class AlloNegative(SlotlessFunction):
    annotation_relevant = False

    communicative_function: Literal[CommunicativeFunction.ALLO_NEGATIVE] = Field(
        title="alloNegative",
        description="Feedback act indicating the sender believes the addressee had trouble processing previous utterance(s).",
        examples=["No no no no no"],
    )


class FeedbackElicitation(SlotlessFunction):
    annotation_relevant = False

    communicative_function: Literal[CommunicativeFunction.FEEDBACK_ELICITATION] = Field(
        title="feedbackElicitation",
        description="Feedback act asking whether the addressee successfully processed previous utterance(s).",
        examples=["Okay?", "Capisce?", "Ja?"],
    )


class TurnAccept(SlotlessFunction):
    annotation_relevant = False

    communicative_function: Literal[CommunicativeFunction.TURN_ACCEPT] = Field(
        title="turnAccept",
        description="Turn-management act signalling willingness to take the speaker role as requested by a previous speaker.",
        examples=['A: "What do you say, Craig?" C: "OK, let me see."'],
    )


class TurnTake(SlotlessFunction):
    annotation_relevant = False

    communicative_function: Literal[CommunicativeFunction.TURN_TAKE] = Field(
        title="turnTake",
        description="Turn-management act taking the currently available speaker role.",
        examples=["Uh... (turn-initial)"],
    )


class TurnGrab(SlotlessFunction):
    annotation_relevant = False

    communicative_function: Literal[CommunicativeFunction.TURN_GRAB] = Field(
        title="turnGrab",
        description="Turn-management act taking the speaker role away from the current speaker.",
        examples=["Hold on", "raised hand as stop signal"],
    )


class TurnAssign(SlotlessFunction):
    annotation_relevant = False

    communicative_function: Literal[CommunicativeFunction.TURN_ASSIGN] = Field(
        title="turnAssign",
        description="Turn-management act passing the speaker role to a designated participant.",
        examples=["Craig?"],
    )


class TurnRelease(SlotlessFunction):
    annotation_relevant = False

    communicative_function: Literal[CommunicativeFunction.TURN_RELEASE] = Field(
        title="turnRelease",
        description="Turn-management act making the speaker role available to others.",
        examples=["declining intonation followed by a pause"],
    )


class TurnKeep(SlotlessFunction):
    annotation_relevant = False

    communicative_function: Literal[CommunicativeFunction.TURN_KEEP] = Field(
        title="turnKeep",
        description="Turn-management act keeping the speaker role.",
        examples=["Uh (not turn-initial)"],
    )


class Stalling(SlotlessFunction):
    annotation_relevant = True

    communicative_function: Literal[CommunicativeFunction.STALLING] = Field(
        title="stalling",
        description="Communicative function of a dialogue act performed in order to have a little more time for constructing his contribution.",
        examples=["Let me see...", "Uh...", "speaking slowly", "We... we went to…"],
    )


class Pausing(SlotlessFunction):
    annotation_relevant = True

    communicative_function: Literal[CommunicativeFunction.PAUSING] = Field(
        title="pausing",
        description="Communicative function of a dialogue act performed in order to suspend the dialogue for a short while.",
        examples=["Just a moment"],
    )


class InteractionStructuring(SlotlessFunction):
    annotation_relevant = False

    communicative_function: Literal[CommunicativeFunction.INTERACTION_STRUCTURING] = (
        Field(
            title="interactionStructuring",
            description="Act explicitly structuring the interaction, e.g. opening/closing a topic or announcing what comes next.",
            examples=["And the windows, we had to replace all the windows"],
        )
    )


class Opening(SlotlessFunction):
    annotation_relevant = False

    communicative_function: Literal[CommunicativeFunction.OPENING] = Field(
        title="opening",
        description="Act indicating the sender is ready and willing to engage in dialogue.",
        examples=["Okay (at the start of a multi-party dialogue)"],
    )


class TopicShift(SlotlessFunction):
    annotation_relevant = False

    communicative_function: Literal[CommunicativeFunction.TOPIC_SHIFT] = Field(
        title="topicShift",
        description="Act indicating the sender will continue on a different topic.",
        examples=["Something else."],
    )


class SelfError(SlotlessFunction):
    annotation_relevant = False

    communicative_function: Literal[CommunicativeFunction.SELF_ERROR] = Field(
        title="selfError",
        description="Act signalling that the sender made a mistake in speaking.",
        examples=["yes oh sorry no..."],
    )


class Retraction(SlotValueFunction):
    annotation_relevant = False

    communicative_function: Literal[CommunicativeFunction.RETRACTION] = Field(
        title="retraction",
        description="Act withdrawing something the sender just said within the same turn.",
        examples=["then we're going to g--"],
    )


class SelfCorrection(SlotValueFunction):
    annotation_relevant = False

    communicative_function: Literal[CommunicativeFunction.SELF_CORRECTION] = Field(
        title="selfCorrection",
        description="Act correcting the sender's own speaking error or improving their own formulation within the same turn.",
        examples=["then we're going to g-- ... turn straight back"],
    )


class Completion(SlotlessFunction):
    annotation_relevant = False

    communicative_function: Literal[CommunicativeFunction.COMPLETION] = Field(
        title="completion",
        description="Act assisting the addressee in completing an utterance.",
        examples=[
            'A: "which should leave us plenty of time to uh... uh" S: "get to Corning"'
        ],
    )


class CorrectMisspeaking(SlotlessFunction):
    annotation_relevant = False

    communicative_function: Literal[CommunicativeFunction.CORRECT_MISSPEAKING] = Field(
        title="correctMisspeaking",
        description="Act correcting part of the addressee's utterance under the assumption that the addressee misspoke.",
        examples=['A: "...pick up the bananas..." S: "to pick up the oranges"'],
    )


class Greeting(SlotlessFunction):
    annotation_relevant = True

    communicative_function: Literal[CommunicativeFunction.GREETING] = Field(
        title="Greeting",
        description="Communicative function of a dialogue act, in order to inform present and aware;",
        examples=["Hello!", "Good morning", "How are you?"],
    )


class InitGreeting(SlotlessFunction):
    annotation_relevant = False

    communicative_function: Literal[CommunicativeFunction.INIT_GREETING] = Field(
        title="initGreeting",
        description="Communicative function of a dialogue act performed by the sender, S, in order to inform the addressee, A, that S is present and aware of A's presence; S puts pressure on A to acknowledge this.",
        examples=["Hello!", "Good morning", "How are you?"],
    )


class ReturnGreeting(SlotlessFunction):
    annotation_relevant = False

    communicative_function: Literal[CommunicativeFunction.RETURN_GREETING] = Field(
        title="returnGreeting",
        description="Communicative function of a dialogue act performed by the sender, S, in order to acknowledge that S is aware of the presence of the addressee, A, and of A having signalled his presence to S.",
        examples=['I: "Schiphol Information, good morning." C: "Good morning"'],
    )


class InitSelfIntroduction(SlotlessFunction):
    annotation_relevant = False

    communicative_function: Literal[CommunicativeFunction.INIT_SELF_INTRODUCTION] = (
        Field(
            title="initSelfIntroduction",
            description="Communicative function of a dialogue act performed by the sender, S, in order to make himself/herself known to the addressee, A; S puts pressure on A to acknowledge this.",
            examples=["Schiphol Information."],
        )
    )


class ReturnSelfIntroduction(SlotlessFunction):
    annotation_relevant = False

    communicative_function: Literal[CommunicativeFunction.RETURN_SELF_INTRODUCTION] = (
        Field(
            title="returnSelfIntroduction",
            description="Self-introduction made in response to another self-introduction.",
            examples=[
                'I: "Schiphol Information, good morning." C: "Good morning, this is De Bruin in Arnhem."'
            ],
        )
    )


class Apology(SlotlessFunction):
    annotation_relevant = False

    communicative_function: Literal[CommunicativeFunction.APOLOGY] = Field(
        title="apology",
        description="Act expressing regret and pressuring the addressee to acknowledge it.",
        examples=["sorry, pick up the oranges"],
    )


class AcceptApology(SlotlessFunction):
    annotation_relevant = False

    communicative_function: Literal[CommunicativeFunction.ACCEPT_APOLOGY] = Field(
        title="acceptApology",
        description="Act mitigating or accepting the addressee's expressed regret.",
        examples=["No problem."],
    )


class Thanking(SlotlessFunction):
    annotation_relevant = False

    communicative_function: Literal[CommunicativeFunction.THANKING] = Field(
        title="thanking",
        description="Act expressing gratitude for an addressee's action.",
        examples=["Thanks a lot.", "Muito obrigado", "Tack so mycket", "Evcharisto"],
    )


class AcceptThanking(SlotlessFunction):
    annotation_relevant = False

    communicative_function: Literal[CommunicativeFunction.ACCEPT_THANKING] = Field(
        title="acceptThanking",
        description="Act mitigating or accepting the addressee's expressed gratitude.",
        examples=["Don't mention it", "De nada", "parakalo"],
    )


class Goodbye(SlotlessFunction):
    annotation_relevant = True

    communicative_function: Literal[CommunicativeFunction.GOODBYE] = Field(
        title="Goodbye",
        description="Communicative function of a dialogue act, in order to signal the current utterance is his last contribution to the dialogue.",
        examples=["Bye bye, see you later"],
    )


class InitGoodbye(SlotlessFunction):
    annotation_relevant = False

    communicative_function: Literal[CommunicativeFunction.GOODBYE] = Field(
        title="initGoodbye",
        description="Communicative function of a dialogue act performed by the sender, S, in order to signal the current utterance is his last contribution to the dialogue; S pressures the addressee, A, to respond with a returnGoodbye act.",
        examples=["Bye bye, see you later"],
    )


class ReturnGoodbye(SlotlessFunction):
    annotation_relevant = False

    communicative_function: Literal[CommunicativeFunction.RETURN_GOODBYE] = Field(
        title="returnGoodbye",
        description="Communicative function of a dialogue act performed by the sender, S, in order to acknowledge his awareness that the addressee, A, has made his last contribution to the dialogue and to signal his agreement to end the dialogue.",
        examples=["Bye bye, see you."],
    )


class Compliment(SlotlessFunction):
    annotation_relevant = False

    communicative_function: Literal[CommunicativeFunction.COMPLIMENT] = Field(
        title="compliment",
        description="Act expressing positive evaluation of the addressee's appearance, qualities, or achievement.",
        examples=["Well done!", "You look great"],
    )


class Congratulation(SlotlessFunction):
    annotation_relevant = False

    communicative_function: Literal[CommunicativeFunction.CONGRATULATION] = Field(
        title="congratulation",
        description="Act expressing pleasure in the addressee's success or good fortune.",
        examples=["Congratulations!"],
    )


class SympathyExpression(SlotlessFunction):
    annotation_relevant = False

    communicative_function: Literal[CommunicativeFunction.SYMPATHY_EXPRESSION] = Field(
        title="sympathyExpression",
        description="Act expressing sympathy about something that happened to the addressee.",
        examples=["I'm sorry to hear that"],
    )


class ContactCheck(SlotlessFunction):
    annotation_relevant = False

    communicative_function: Literal[CommunicativeFunction.CONTACT_CHECK] = Field(
        title="contactCheck",
        description="Contact-management act verifying that the addressee is ready or available to communicate.",
        examples=["Yes?", "Hello?"],
    )


class ContactIndication(SlotlessFunction):
    annotation_relevant = False

    communicative_function: Literal[CommunicativeFunction.CONTACT_INDICATION] = Field(
        title="contactIndication",
        description="Communicative function of a dialogue act performed by the sender, S, in order to make it known to the addressee, A, that S is ready to communicate with A.",
        examples=["Yes", "Oh hi", "Grüß Gott", "Hello"],
    )


COMMUNICATIVE_FUNCTION_TYPE_BY_VALUE = {
    CommunicativeFunction.QUESTION: Question,
    CommunicativeFunction.PROPOSITIONAL_QUESTION: PropositionalQuestion,
    CommunicativeFunction.CHECK_QUESTION: CheckQuestion,
    CommunicativeFunction.SET_QUESTION: SetQuestion,
    CommunicativeFunction.CHOICE_QUESTION: ChoiceQuestion,
    CommunicativeFunction.TEST_QUESTION: TestQuestion,
    CommunicativeFunction.INFORM: Inform,
    CommunicativeFunction.AGREEMENT: Agreement,
    CommunicativeFunction.DISAGREEMENT: Disagreement,
    CommunicativeFunction.CORRECTION: Correction,
    CommunicativeFunction.ANSWER: Answer,
    CommunicativeFunction.CONFIRM: Confirm,
    CommunicativeFunction.DISCONFIRM: Disconfirm,
    CommunicativeFunction.OFFER: Offer,
    CommunicativeFunction.PROMISE: Promise,
    CommunicativeFunction.ADDRESS_REQUEST: AddressRequest,
    CommunicativeFunction.ACCEPT_REQUEST: AcceptRequest,
    CommunicativeFunction.DECLINE_REQUEST: DeclineRequest,
    CommunicativeFunction.ADDRESS_SUGGEST: AddressSuggest,
    CommunicativeFunction.ACCEPT_SUGGEST: AcceptSuggest,
    CommunicativeFunction.DECLINE_SUGGEST: DeclineSuggest,
    CommunicativeFunction.REQUEST: Request,
    CommunicativeFunction.INSTRUCT: Instruct,
    CommunicativeFunction.SUGGEST: Suggest,
    CommunicativeFunction.ADDRESS_OFFER: AddressOffer,
    CommunicativeFunction.ACCEPT_OFFER: AcceptOffer,
    CommunicativeFunction.DECLINE_OFFER: DeclineOffer,
    CommunicativeFunction.AUTO_POSITIVE: AutoPositive,
    CommunicativeFunction.AUTO_NEGATIVE: AutoNegative,
    CommunicativeFunction.ALLO_POSITIVE: AlloPositive,
    CommunicativeFunction.ALLO_NEGATIVE: AlloNegative,
    CommunicativeFunction.FEEDBACK_ELICITATION: FeedbackElicitation,
    CommunicativeFunction.TURN_ACCEPT: TurnAccept,
    CommunicativeFunction.TURN_TAKE: TurnTake,
    CommunicativeFunction.TURN_GRAB: TurnGrab,
    CommunicativeFunction.TURN_ASSIGN: TurnAssign,
    CommunicativeFunction.TURN_RELEASE: TurnRelease,
    CommunicativeFunction.TURN_KEEP: TurnKeep,
    CommunicativeFunction.STALLING: Stalling,
    CommunicativeFunction.PAUSING: Pausing,
    CommunicativeFunction.INTERACTION_STRUCTURING: InteractionStructuring,
    CommunicativeFunction.OPENING: Opening,
    CommunicativeFunction.TOPIC_SHIFT: TopicShift,
    CommunicativeFunction.SELF_ERROR: SelfError,
    CommunicativeFunction.RETRACTION: Retraction,
    CommunicativeFunction.SELF_CORRECTION: SelfCorrection,
    CommunicativeFunction.COMPLETION: Completion,
    CommunicativeFunction.CORRECT_MISSPEAKING: CorrectMisspeaking,
    CommunicativeFunction.INIT_GREETING: InitGreeting,
    CommunicativeFunction.RETURN_GREETING: ReturnGreeting,
    CommunicativeFunction.INIT_SELF_INTRODUCTION: InitSelfIntroduction,
    CommunicativeFunction.RETURN_SELF_INTRODUCTION: ReturnSelfIntroduction,
    CommunicativeFunction.APOLOGY: Apology,
    CommunicativeFunction.ACCEPT_APOLOGY: AcceptApology,
    CommunicativeFunction.THANKING: Thanking,
    CommunicativeFunction.ACCEPT_THANKING: AcceptThanking,
    CommunicativeFunction.INIT_GOODBYE: InitGoodbye,
    CommunicativeFunction.RETURN_GOODBYE: ReturnGoodbye,
    CommunicativeFunction.COMPLIMENT: Compliment,
    CommunicativeFunction.CONGRATULATION: Congratulation,
    CommunicativeFunction.SYMPATHY_EXPRESSION: SympathyExpression,
    CommunicativeFunction.CONTACT_CHECK: ContactCheck,
    CommunicativeFunction.CONTACT_INDICATION: ContactIndication,
    CommunicativeFunction.GOODBYE: Goodbye,
    CommunicativeFunction.GREETING: Greeting,
}
FunctionLabel = Annotated[
    Union[
        tuple(
            function_type
            for function_type in COMMUNICATIVE_FUNCTION_TYPE_BY_VALUE.values()
            if function_type.annotation_relevant
        )
    ],
    Field(discriminator="communicative_function"),
]  # pyrefly: ignore [invalid-annotation]


if __name__ == "__main__":
    print(Question.model_json_schema()["properties"]["communicative_function"])
    print(inline_json_schema_refs(Question.model_json_schema()))
    print(inline_json_schema_refs(Inform.model_json_schema()))


# Old Code!
#
#
class SlotMode(StrEnum):
    SLOTLESS = "slotless"
    SLOT_ONLY = "slot_only"
    CHOICE = "choice"
    VALUE = "value"


class CommunicativeFunctionConfig(BaseModel):
    """Configuration for a communicative function with slot mode and annotation metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    slot_mode: SlotMode = Field(
        description="Type of slots this function requires: slotless (no annotation), slot_only (slot required), choice (choice between alternatives), or value (value required)"
    )
    description: str = Field(
        min_length=1,
        description="Human-readable description of this communicative function",
    )
    examples: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Example utterances that demonstrate this communicative function",
    )
    annotation_relevant: bool = Field(
        default=True,
        description="Whether this function is relevant for annotation tasks",
    )
    policy_relevant: bool = Field(
        default=False, description="Whether this function is relevant for dialog policy"
    )


# Internal data structure for configs
_COMMUNICATIVE_FUNCTION_CONFIGS_DATA: dict[
    CommunicativeFunction, CommunicativeFunctionConfig
] = {
    CommunicativeFunction.QUESTION: CommunicativeFunctionConfig(
        slot_mode=SlotMode.SLOTLESS,
        description="Generic information-seeking act: the sender tries to obtain information they assume the addressee possesses.",
        examples=("And so?",),
        annotation_relevant=True,
        policy_relevant=False,
    ),
    CommunicativeFunction.PROPOSITIONAL_QUESTION: CommunicativeFunctionConfig(
        slot_mode=SlotMode.SLOT_ONLY,
        description="Question asking whether a proposition is true.",
        examples=("Does the meeting start at  ten?",),
        annotation_relevant=False,
        policy_relevant=True,
    ),
    CommunicativeFunction.CHECK_QUESTION: CommunicativeFunctionConfig(
        slot_mode=SlotMode.SLOT_ONLY,
        description="Question asking for confirmation of a proposition the sender tentatively believes to be true.",
        examples=("The meeting starts at ten, right?",),
        annotation_relevant=False,
        policy_relevant=True,
    ),
    CommunicativeFunction.SET_QUESTION: CommunicativeFunctionConfig(
        slot_mode=SlotMode.SLOT_ONLY,
        description="Question asking which value, entity, time, place, person, reason, or other set member satisfies a property.",
        examples=("What time does the meeting start?", "How far is it to the station?"),
        annotation_relevant=False,
        policy_relevant=True,
    ),
    CommunicativeFunction.CHOICE_QUESTION: CommunicativeFunctionConfig(
        slot_mode=SlotMode.CHOICE,
        description="Question asking which one of an explicit set of alternative propositions is true.",
        examples=(
            "Should the telephone cable go in the telephone line slot or in the external line slot?",
        ),
        annotation_relevant=False,
        policy_relevant=True,
    ),
    CommunicativeFunction.TEST_QUESTION: CommunicativeFunctionConfig(
        slot_mode=SlotMode.SLOTLESS,
        description="Question where the sender already knows the answer and asks to test whether the addressee knows it.",
        examples=(),
        annotation_relevant=False,
        policy_relevant=False,
    ),
    CommunicativeFunction.INFORM: CommunicativeFunctionConfig(
        slot_mode=SlotMode.VALUE,
        description="Information-providing act: the sender makes information available to the addressee and presents it as correct.",
        examples=("The 6.34 to Breda leaves from platform 2.",),
        annotation_relevant=True,
        policy_relevant=True,
    ),
    CommunicativeFunction.AGREEMENT: CommunicativeFunctionConfig(
        slot_mode=SlotMode.SLOTLESS,
        description="Act expressing that the sender agrees with or accepts prior information as true.",
        examples=("Exactly", "Precies!", "Netop!"),
        annotation_relevant=False,
        policy_relevant=True,
    ),
    CommunicativeFunction.DISAGREEMENT: CommunicativeFunctionConfig(
        slot_mode=SlotMode.SLOTLESS,
        description="Act expressing that the sender rejects prior information as false.",
        examples=("uh… no",),
        annotation_relevant=False,
        policy_relevant=True,
    ),
    CommunicativeFunction.CORRECTION: CommunicativeFunctionConfig(
        slot_mode=SlotMode.VALUE,
        description="Act that rejects information assumed by the addressee and provides replacement information.",
        examples=("To Montreal, not to Ottawa",),
        annotation_relevant=True,
        policy_relevant=True,
    ),
    CommunicativeFunction.ANSWER: CommunicativeFunctionConfig(
        slot_mode=SlotMode.VALUE,
        description="Information-providing act that supplies information the addressee wanted to know, typically in response to a question.",
        examples=("send error document ready",),
        annotation_relevant=False,
        policy_relevant=True,
    ),
    CommunicativeFunction.CONFIRM: CommunicativeFunctionConfig(
        slot_mode=SlotMode.SLOTLESS,
        description="Answer confirming that a proposition being checked is true.",
        examples=("Indeed",),
        annotation_relevant=False,
        policy_relevant=True,
    ),
    CommunicativeFunction.DISCONFIRM: CommunicativeFunctionConfig(
        slot_mode=SlotMode.SLOTLESS,
        description="Answer rejecting a proposition being checked as false.",
        examples=("si", "jo", "toch niet", "toch wel", "doch"),
        annotation_relevant=False,
        policy_relevant=True,
    ),
    CommunicativeFunction.OFFER: CommunicativeFunctionConfig(
        slot_mode=SlotMode.SLOTLESS,
        description="Commissive act where the sender conditionally commits to perform an action if the addressee consents.",
        examples=("I will look that up for you",),
        annotation_relevant=False,
        policy_relevant=True,
    ),
    CommunicativeFunction.PROMISE: CommunicativeFunctionConfig(
        slot_mode=SlotMode.SLOTLESS,
        description="Commissive act where the sender commits to perform an action believed to benefit the addressee.",
        examples=("Shall I begin?", "Would you like to have some coffee?"),
        annotation_relevant=False,
        policy_relevant=True,
    ),
    CommunicativeFunction.ADDRESS_REQUEST: CommunicativeFunctionConfig(
        slot_mode=SlotMode.SLOTLESS,
        description="Response indicating the sender is considering performing an action that was requested, possibly conditionally.",
        examples=('A: "Give me the gun." S: "If you push the bag to me"',),
        annotation_relevant=False,
        policy_relevant=True,
    ),
    CommunicativeFunction.ACCEPT_REQUEST: CommunicativeFunctionConfig(
        slot_mode=SlotMode.SLOTLESS,
        description="Response committing to perform an action that was requested.",
        examples=('A: "Could you close the door please?" B: "Sure."',),
        annotation_relevant=False,
        policy_relevant=True,
    ),
    CommunicativeFunction.DECLINE_REQUEST: CommunicativeFunctionConfig(
        slot_mode=SlotMode.SLOTLESS,
        description="Response committing not to perform an action that was requested.",
        examples=("Not now.",),
        annotation_relevant=False,
        policy_relevant=True,
    ),
    CommunicativeFunction.ADDRESS_SUGGEST: CommunicativeFunctionConfig(
        slot_mode=SlotMode.SLOTLESS,
        description="Response indicating the sender is considering a suggested action, possibly under conditions.",
        examples=(
            'A: "Let\'s go together." S: "Only if we\'re in full agreement about how to proceed when we get there."',
        ),
        annotation_relevant=False,
        policy_relevant=True,
    ),
    CommunicativeFunction.ACCEPT_SUGGEST: CommunicativeFunctionConfig(
        slot_mode=SlotMode.SLOTLESS,
        description="Response committing to perform or go along with a suggested action.",
        examples=('A: "Shall we go and have a look around?" B: "Let\'s do so."',),
        annotation_relevant=False,
        policy_relevant=True,
    ),
    CommunicativeFunction.DECLINE_SUGGEST: CommunicativeFunctionConfig(
        slot_mode=SlotMode.SLOTLESS,
        description="Response indicating the sender will not perform or go along with a suggested action.",
        examples=("I'd rather not.",),
        annotation_relevant=False,
        policy_relevant=True,
    ),
    CommunicativeFunction.REQUEST: CommunicativeFunctionConfig(
        slot_mode=SlotMode.SLOTLESS,
        description="Directive act asking the addressee to perform an action, conditional on the addressee consenting.",
        examples=("Please turn to page five", "Don't do this ever again, please"),
        annotation_relevant=False,
        policy_relevant=False,
    ),
    CommunicativeFunction.INSTRUCT: CommunicativeFunctionConfig(
        slot_mode=SlotMode.SLOTLESS,
        description="Directive act making the addressee feel obliged to perform an action.",
        examples=("Go right round until you get to just above that.",),
        annotation_relevant=False,
        policy_relevant=True,
    ),
    CommunicativeFunction.SUGGEST: CommunicativeFunctionConfig(
        slot_mode=SlotMode.SLOTLESS,
        description="Directive act inviting the addressee to consider an action believed to be in their interest.",
        examples=("Let's wait for the speaker to finish.",),
        annotation_relevant=False,
        policy_relevant=True,
    ),
    CommunicativeFunction.ADDRESS_OFFER: CommunicativeFunctionConfig(
        slot_mode=SlotMode.SLOTLESS,
        description="Response indicating the sender is considering whether the addressee should perform an offered action.",
        examples=("Yes please", "Je vous en prie"),
        annotation_relevant=False,
        policy_relevant=True,
    ),
    CommunicativeFunction.ACCEPT_OFFER: CommunicativeFunctionConfig(
        slot_mode=SlotMode.SLOTLESS,
        description="Response indicating the sender wants the addressee to perform the offered action.",
        examples=("Yes please", "Je vous en prie", "Graag", "Bitte"),
        annotation_relevant=False,
        policy_relevant=True,
    ),
    CommunicativeFunction.DECLINE_OFFER: CommunicativeFunctionConfig(
        slot_mode=SlotMode.SLOTLESS,
        description="Response indicating the sender does not want the addressee to perform the offered action.",
        examples=("No thank you", "Nej tak", "Non merci"),
        annotation_relevant=False,
        policy_relevant=True,
    ),
    CommunicativeFunction.AUTO_POSITIVE: CommunicativeFunctionConfig(
        slot_mode=SlotMode.SLOTLESS,
        description="Feedback act indicating the sender successfully processed previous utterance(s).",
        examples=("Uh-huh", "Okay", "nodding", "Yes"),
        annotation_relevant=False,
        policy_relevant=True,
    ),
    CommunicativeFunction.AUTO_NEGATIVE: CommunicativeFunctionConfig(
        slot_mode=SlotMode.SLOTLESS,
        description="Feedback act indicating the sender had trouble processing previous utterance(s).",
        examples=("I beg your pardon", "Como?"),
        annotation_relevant=False,
        policy_relevant=True,
    ),
    CommunicativeFunction.ALLO_POSITIVE: CommunicativeFunctionConfig(
        slot_mode=SlotMode.SLOTLESS,
        description="Feedback act indicating the sender believes the addressee successfully processed previous utterance(s).",
        examples=("Correct", "Right"),
        annotation_relevant=False,
        policy_relevant=True,
    ),
    CommunicativeFunction.ALLO_NEGATIVE: CommunicativeFunctionConfig(
        slot_mode=SlotMode.SLOTLESS,
        description="Feedback act indicating the sender believes the addressee had trouble processing previous utterance(s).",
        examples=("No no no no no",),
        annotation_relevant=False,
        policy_relevant=True,
    ),
    CommunicativeFunction.FEEDBACK_ELICITATION: CommunicativeFunctionConfig(
        slot_mode=SlotMode.SLOTLESS,
        description="Feedback act asking whether the addressee successfully processed previous utterance(s).",
        examples=("Okay?", "Capisce?", "Ja?"),
        annotation_relevant=False,
        policy_relevant=True,
    ),
    CommunicativeFunction.TURN_ACCEPT: CommunicativeFunctionConfig(
        slot_mode=SlotMode.SLOTLESS,
        description="Turn-management act signalling willingness to take the speaker role as requested by a previous speaker.",
        examples=('A: "What do you say, Craig?" C: "OK, let me see."',),
        annotation_relevant=False,
        policy_relevant=False,
    ),
    CommunicativeFunction.TURN_TAKE: CommunicativeFunctionConfig(
        slot_mode=SlotMode.SLOTLESS,
        description="Turn-management act taking the currently available speaker role.",
        examples=("Uh... (turn-initial)",),
        annotation_relevant=False,
        policy_relevant=False,
    ),
    CommunicativeFunction.TURN_GRAB: CommunicativeFunctionConfig(
        slot_mode=SlotMode.SLOTLESS,
        description="Turn-management act taking the speaker role away from the current speaker.",
        examples=("Hold on", "raised hand as stop signal"),
        annotation_relevant=False,
        policy_relevant=False,
    ),
    CommunicativeFunction.TURN_ASSIGN: CommunicativeFunctionConfig(
        slot_mode=SlotMode.SLOTLESS,
        description="Turn-management act passing the speaker role to a designated participant.",
        examples=("Craig?",),
        annotation_relevant=False,
        policy_relevant=False,
    ),
    CommunicativeFunction.TURN_RELEASE: CommunicativeFunctionConfig(
        slot_mode=SlotMode.SLOTLESS,
        description="Turn-management act making the speaker role available to others.",
        examples=("declining intonation followed by a pause",),
        annotation_relevant=False,
        policy_relevant=False,
    ),
    CommunicativeFunction.TURN_KEEP: CommunicativeFunctionConfig(
        slot_mode=SlotMode.SLOTLESS,
        description="Turn-management act keeping the speaker role.",
        examples=("Uh (not turn-initial)",),
        annotation_relevant=False,
        policy_relevant=False,
    ),
    CommunicativeFunction.STALLING: CommunicativeFunctionConfig(
        slot_mode=SlotMode.SLOTLESS,
        description="Time-management act used to gain a little time while constructing the contribution.",
        examples=("Let me see...", "Uh...", "speaking slowly", "We... we went to…"),
        annotation_relevant=False,
        policy_relevant=False,
    ),
    CommunicativeFunction.PAUSING: CommunicativeFunctionConfig(
        slot_mode=SlotMode.SLOTLESS,
        description="Time-management act used to suspend the dialogue for a short while.",
        examples=("Just a moment",),
        annotation_relevant=False,
        policy_relevant=False,
    ),
    CommunicativeFunction.INTERACTION_STRUCTURING: CommunicativeFunctionConfig(
        slot_mode=SlotMode.SLOTLESS,
        description="Act explicitly structuring the interaction, e.g. opening/closing a topic or announcing what comes next.",
        examples=("And the windows, we had to replace all the windows",),
        annotation_relevant=False,
        policy_relevant=True,
    ),
    CommunicativeFunction.OPENING: CommunicativeFunctionConfig(
        slot_mode=SlotMode.SLOTLESS,
        description="Act indicating the sender is ready and willing to engage in dialogue.",
        examples=("Okay (at the start of a multi-party dialogue)",),
        annotation_relevant=False,
        policy_relevant=True,
    ),
    CommunicativeFunction.TOPIC_SHIFT: CommunicativeFunctionConfig(
        slot_mode=SlotMode.SLOTLESS,
        description="Act indicating the sender will continue on a different topic.",
        examples=("Something else.",),
        annotation_relevant=False,
        policy_relevant=True,
    ),
    CommunicativeFunction.SELF_ERROR: CommunicativeFunctionConfig(
        slot_mode=SlotMode.SLOTLESS,
        description="Act signalling that the sender made a mistake in speaking.",
        examples=("yes oh sorry no...",),
        annotation_relevant=False,
        policy_relevant=True,
    ),
    CommunicativeFunction.RETRACTION: CommunicativeFunctionConfig(
        slot_mode=SlotMode.VALUE,
        description="Act withdrawing something the sender just said within the same turn.",
        examples=("then we're going to g--",),
        annotation_relevant=False,
        policy_relevant=True,
    ),
    CommunicativeFunction.SELF_CORRECTION: CommunicativeFunctionConfig(
        slot_mode=SlotMode.VALUE,
        description="Act correcting the sender's own speaking error or improving their own formulation within the same turn.",
        examples=("then we're going to g-- ... turn straight back",),
        annotation_relevant=False,
        policy_relevant=True,
    ),
    CommunicativeFunction.COMPLETION: CommunicativeFunctionConfig(
        slot_mode=SlotMode.SLOTLESS,
        description="Act assisting the addressee in completing an utterance.",
        examples=(
            'A: "which should leave us plenty of time to uh... uh" S: "get to Corning"',
        ),
        annotation_relevant=False,
        policy_relevant=False,
    ),
    CommunicativeFunction.CORRECT_MISSPEAKING: CommunicativeFunctionConfig(
        slot_mode=SlotMode.SLOTLESS,
        description="Act correcting part of the addressee's utterance under the assumption that the addressee misspoke.",
        examples=('A: "...pick up the bananas..." S: "to pick up the oranges"',),
        annotation_relevant=False,
        policy_relevant=False,
    ),
    CommunicativeFunction.INIT_GREETING: CommunicativeFunctionConfig(
        slot_mode=SlotMode.SLOTLESS,
        description="Initial greeting signalling the sender's presence and awareness of the addressee.",
        examples=("Hello!", "Good morning", "How are you?"),
        annotation_relevant=False,
        policy_relevant=True,
    ),
    CommunicativeFunction.RETURN_GREETING: CommunicativeFunctionConfig(
        slot_mode=SlotMode.SLOTLESS,
        description="Greeting returned in response to an initial greeting.",
        examples=('I: "Schiphol Information, good morning." C: "Good morning"',),
        annotation_relevant=False,
        policy_relevant=True,
    ),
    CommunicativeFunction.INIT_SELF_INTRODUCTION: CommunicativeFunctionConfig(
        slot_mode=SlotMode.SLOTLESS,
        description="Initial self-introduction making the sender known to the addressee.",
        examples=("Schiphol Information.",),
        annotation_relevant=False,
        policy_relevant=True,
    ),
    CommunicativeFunction.RETURN_SELF_INTRODUCTION: CommunicativeFunctionConfig(
        slot_mode=SlotMode.SLOTLESS,
        description="Self-introduction made in response to another self-introduction.",
        examples=(
            'I: "Schiphol Information, good morning." C: "Good morning, this is De Bruin in Arnhem."',
        ),
        annotation_relevant=False,
        policy_relevant=True,
    ),
    CommunicativeFunction.APOLOGY: CommunicativeFunctionConfig(
        slot_mode=SlotMode.SLOTLESS,
        description="Act expressing regret and pressuring the addressee to acknowledge it.",
        examples=("sorry, pick up the oranges",),
        annotation_relevant=False,
        policy_relevant=True,
    ),
    CommunicativeFunction.ACCEPT_APOLOGY: CommunicativeFunctionConfig(
        slot_mode=SlotMode.SLOTLESS,
        description="Act mitigating or accepting the addressee's expressed regret.",
        examples=("No problem.",),
        annotation_relevant=False,
        policy_relevant=True,
    ),
    CommunicativeFunction.THANKING: CommunicativeFunctionConfig(
        slot_mode=SlotMode.SLOTLESS,
        description="Act expressing gratitude for an addressee's action.",
        examples=("Thanks a lot.", "Muito obrigado", "Tack so mycket", "Evcharisto"),
        annotation_relevant=False,
        policy_relevant=True,
    ),
    CommunicativeFunction.ACCEPT_THANKING: CommunicativeFunctionConfig(
        slot_mode=SlotMode.SLOTLESS,
        description="Act mitigating or accepting the addressee's expressed gratitude.",
        examples=("Don't mention it", "De nada", "parakalo"),
        annotation_relevant=False,
        policy_relevant=True,
    ),
    CommunicativeFunction.INIT_GOODBYE: CommunicativeFunctionConfig(
        slot_mode=SlotMode.SLOTLESS,
        description="Initial goodbye signalling the sender's current utterance is their last contribution.",
        examples=("Bye bye, see you later",),
        annotation_relevant=False,
        policy_relevant=True,
    ),
    CommunicativeFunction.RETURN_GOODBYE: CommunicativeFunctionConfig(
        slot_mode=SlotMode.SLOTLESS,
        description="Goodbye returned in response to an initial goodbye, acknowledging the end of the dialogue.",
        examples=("Bye bye, see you.",),
        annotation_relevant=False,
        policy_relevant=True,
    ),
    CommunicativeFunction.COMPLIMENT: CommunicativeFunctionConfig(
        slot_mode=SlotMode.SLOTLESS,
        description="Act expressing positive evaluation of the addressee's appearance, qualities, or achievement.",
        examples=("Well done!", "You look great"),
        annotation_relevant=False,
        policy_relevant=False,
    ),
    CommunicativeFunction.CONGRATULATION: CommunicativeFunctionConfig(
        slot_mode=SlotMode.SLOTLESS,
        description="Act expressing pleasure in the addressee's success or good fortune.",
        examples=("Congratulations!",),
        annotation_relevant=False,
        policy_relevant=False,
    ),
    CommunicativeFunction.SYMPATHY_EXPRESSION: CommunicativeFunctionConfig(
        slot_mode=SlotMode.SLOTLESS,
        description="Act expressing sympathy about something that happened to the addressee.",
        examples=("I'm sorry to hear that",),
        annotation_relevant=False,
        policy_relevant=False,
    ),
    CommunicativeFunction.CONTACT_CHECK: CommunicativeFunctionConfig(
        slot_mode=SlotMode.SLOTLESS,
        description="Contact-management act verifying that the addressee is ready or available to communicate.",
        examples=("Yes?", "Hello?"),
        annotation_relevant=False,
        policy_relevant=True,
    ),
    CommunicativeFunction.CONTACT_INDICATION: CommunicativeFunctionConfig(
        slot_mode=SlotMode.SLOTLESS,
        description="Contact-management act indicating the sender is ready or available to communicate.",
        examples=("Yes", "Oh hi"),
        annotation_relevant=False,
        policy_relevant=True,
    ),
}

COMMUNICATIVE_FUNCTION_CONFIGS: dict[
    CommunicativeFunction, CommunicativeFunctionConfig
] = _COMMUNICATIVE_FUNCTION_CONFIGS_DATA
