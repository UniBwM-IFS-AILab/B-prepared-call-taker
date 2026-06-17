from __future__ import annotations

from enum import StrEnum

from markdown_it.common.normalize_url import GOOD_DATA_RE


class Dimension(StrEnum):
    """Dimensions of ISO 24617-2:2020."""

    TASK = "task"
    TASK_MANAGEMENT = "taskManagement"
    AUTO_FEEDBACK = "autoFeedback"
    ALLO_FEEDBACK = "alloFeedback"
    TURN_MANAGEMENT = "turnManagement"
    TIME_MANAGEMENT = "timeManagement"
    DISCOURSE_STRUCTURING = "discourseStructuring"
    OWN_COMMUNICATION_MANAGEMENT = "ownCommunicationManagement"
    PARTNER_COMMUNICATION_MANAGEMENT = "partnerCommunicationManagement"
    SOCIAL_OBLIGATIONS_MANAGEMENT = "socialObligationsManagement"
    CONTACT_MANAGEMENT = "contactManagement"


class CustomFunction(StrEnum):
    GOODBYE = "goodbye"
    GREETING = "greeting"


class CommunicativeFunction(StrEnum):
    """Communicative functions of ISO 24617-2:2020 C.3."""

    QUESTION = "question"
    PROPOSITIONAL_QUESTION = "propositionalQuestion"
    CHECK_QUESTION = "checkQuestion"
    SET_QUESTION = "setQuestion"
    CHOICE_QUESTION = "choiceQuestion"
    TEST_QUESTION = "testQuestion"
    INFORM = "inform"
    AGREEMENT = "agreement"
    DISAGREEMENT = "disagreement"
    CORRECTION = "correction"
    ANSWER = "answer"
    CONFIRM = "confirm"
    DISCONFIRM = "disconfirm"
    OFFER = "offer"
    PROMISE = "promise"
    ADDRESS_REQUEST = "addressRequest"
    ACCEPT_REQUEST = "acceptRequest"
    DECLINE_REQUEST = "declineRequest"
    ADDRESS_SUGGEST = "addressSuggest"
    ACCEPT_SUGGEST = "acceptSuggest"
    DECLINE_SUGGEST = "declineSuggest"
    REQUEST = "request"
    INSTRUCT = "instruct"
    SUGGEST = "suggest"
    ADDRESS_OFFER = "addressOffer"
    ACCEPT_OFFER = "acceptOffer"
    DECLINE_OFFER = "declineOffer"
    AUTO_POSITIVE = "autoPositive"
    AUTO_NEGATIVE = "autoNegative"
    ALLO_POSITIVE = "alloPositive"
    ALLO_NEGATIVE = "alloNegative"
    FEEDBACK_ELICITATION = "feedbackElicitation"
    TURN_ACCEPT = "turnAccept"
    TURN_TAKE = "turnTake"
    TURN_GRAB = "turnGrab"
    TURN_ASSIGN = "turnAssign"
    TURN_RELEASE = "turnRelease"
    TURN_KEEP = "turnKeep"
    STALLING = "stalling"
    PAUSING = "pausing"
    INTERACTION_STRUCTURING = "interactionStructuring"
    OPENING = "opening"
    TOPIC_SHIFT = "topicShift"
    SELF_ERROR = "selfError"
    RETRACTION = "retraction"
    SELF_CORRECTION = "selfCorrection"
    COMPLETION = "completion"
    CORRECT_MISSPEAKING = "correctMisspeaking"
    INIT_GREETING = "initGreeting"
    RETURN_GREETING = "returnGreeting"
    INIT_SELF_INTRODUCTION = "initSelfIntroduction"
    RETURN_SELF_INTRODUCTION = "returnSelfIntroduction"
    APOLOGY = "apology"
    ACCEPT_APOLOGY = "acceptApology"
    THANKING = "thanking"
    ACCEPT_THANKING = "acceptThanking"
    INIT_GOODBYE = "initGoodbye"
    RETURN_GOODBYE = "returnGoodbye"
    COMPLIMENT = "compliment"
    CONGRATULATION = "congratulation"
    SYMPATHY_EXPRESSION = "sympathyExpression"
    CONTACT_CHECK = "contactCheck"
    CONTACT_INDICATION = "contactIndication"

    GOODBYE = "goodbye"
    GREETING = "greeting"

    @property
    def source_dimension(self) -> Dimension | None:
        return _COMMUNICATIVE_FUNCTION_DIMENSIONS.get(self)

    @property
    def is_general_purpose(self) -> bool:
        return self.source_dimension is None

    @property
    def is_dimension_specific(self) -> bool:
        return self.source_dimension is not None


_COMMUNICATIVE_FUNCTION_DIMENSIONS: dict[CommunicativeFunction, Dimension] = {
    CommunicativeFunction.AUTO_POSITIVE: Dimension.AUTO_FEEDBACK,
    CommunicativeFunction.AUTO_NEGATIVE: Dimension.AUTO_FEEDBACK,
    CommunicativeFunction.ALLO_POSITIVE: Dimension.ALLO_FEEDBACK,
    CommunicativeFunction.ALLO_NEGATIVE: Dimension.ALLO_FEEDBACK,
    CommunicativeFunction.FEEDBACK_ELICITATION: Dimension.ALLO_FEEDBACK,
    CommunicativeFunction.TURN_ACCEPT: Dimension.TURN_MANAGEMENT,
    CommunicativeFunction.TURN_TAKE: Dimension.TURN_MANAGEMENT,
    CommunicativeFunction.TURN_GRAB: Dimension.TURN_MANAGEMENT,
    CommunicativeFunction.TURN_ASSIGN: Dimension.TURN_MANAGEMENT,
    CommunicativeFunction.TURN_RELEASE: Dimension.TURN_MANAGEMENT,
    CommunicativeFunction.TURN_KEEP: Dimension.TURN_MANAGEMENT,
    CommunicativeFunction.STALLING: Dimension.TIME_MANAGEMENT,
    CommunicativeFunction.PAUSING: Dimension.TIME_MANAGEMENT,
    CommunicativeFunction.INTERACTION_STRUCTURING: Dimension.DISCOURSE_STRUCTURING,
    CommunicativeFunction.OPENING: Dimension.DISCOURSE_STRUCTURING,
    CommunicativeFunction.TOPIC_SHIFT: Dimension.DISCOURSE_STRUCTURING,
    CommunicativeFunction.SELF_ERROR: Dimension.OWN_COMMUNICATION_MANAGEMENT,
    CommunicativeFunction.RETRACTION: Dimension.OWN_COMMUNICATION_MANAGEMENT,
    CommunicativeFunction.SELF_CORRECTION: Dimension.OWN_COMMUNICATION_MANAGEMENT,
    CommunicativeFunction.COMPLETION: Dimension.PARTNER_COMMUNICATION_MANAGEMENT,
    CommunicativeFunction.CORRECT_MISSPEAKING: Dimension.PARTNER_COMMUNICATION_MANAGEMENT,
    CommunicativeFunction.INIT_GREETING: Dimension.SOCIAL_OBLIGATIONS_MANAGEMENT,
    CommunicativeFunction.RETURN_GREETING: Dimension.SOCIAL_OBLIGATIONS_MANAGEMENT,
    CommunicativeFunction.INIT_SELF_INTRODUCTION: Dimension.SOCIAL_OBLIGATIONS_MANAGEMENT,
    CommunicativeFunction.RETURN_SELF_INTRODUCTION: Dimension.SOCIAL_OBLIGATIONS_MANAGEMENT,
    CommunicativeFunction.APOLOGY: Dimension.SOCIAL_OBLIGATIONS_MANAGEMENT,
    CommunicativeFunction.ACCEPT_APOLOGY: Dimension.SOCIAL_OBLIGATIONS_MANAGEMENT,
    CommunicativeFunction.THANKING: Dimension.SOCIAL_OBLIGATIONS_MANAGEMENT,
    CommunicativeFunction.ACCEPT_THANKING: Dimension.SOCIAL_OBLIGATIONS_MANAGEMENT,
    CommunicativeFunction.INIT_GOODBYE: Dimension.SOCIAL_OBLIGATIONS_MANAGEMENT,
    CommunicativeFunction.RETURN_GOODBYE: Dimension.SOCIAL_OBLIGATIONS_MANAGEMENT,
    CommunicativeFunction.COMPLIMENT: Dimension.SOCIAL_OBLIGATIONS_MANAGEMENT,
    CommunicativeFunction.CONGRATULATION: Dimension.SOCIAL_OBLIGATIONS_MANAGEMENT,
    CommunicativeFunction.SYMPATHY_EXPRESSION: Dimension.SOCIAL_OBLIGATIONS_MANAGEMENT,
    CommunicativeFunction.CONTACT_CHECK: Dimension.CONTACT_MANAGEMENT,
    CommunicativeFunction.CONTACT_INDICATION: Dimension.CONTACT_MANAGEMENT,
}
