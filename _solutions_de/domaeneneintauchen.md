---
title: Domäneneintauchen
description: Entwickler zur Beobachtung der tatsächlichen Arbeit schicken, die das
  System unterstützt, sodass Anforderungen verstanden statt nur transkribiert werden.
category:
- Requirements
- Team
- Communication
problems:
- complex-domain-model
- inadequate-requirements-gathering
- requirements-ambiguity
- knowledge-gaps
- feedback-isolation
- suboptimal-solutions
- incomplete-knowledge
- reduced-feature-quality
- eager-to-please-stakeholders
- stakeholder-dissatisfaction
- legacy-business-logic-extraction-difficulty
- negative-user-feedback
- declining-business-metrics
- feature-factory
- frequent-changes-to-requirements
- product-direction-chaos
- stakeholder-frustration
- process-software-misfit
- reimplemented-standard-functionality
layout: solution
lang: de
en_slug: domain-immersion
related_solutions:
- slug: on-site-customer
  similarity: 0.65
- slug: requirements-analysis
  similarity: 0.65
- slug: code-reading-sessions
  similarity: 0.65
- slug: evolutionary-requirements-development
  similarity: 0.65
- slug: pair-and-mob-programming
  similarity: 0.6
- slug: exploratory-testing
  similarity: 0.6
---

## Description

Domäneneintauchen bedeutet, dass Entwickler Zeit dort verbringen, wo die vom System unterstützte Arbeit tatsächlich geschieht — mit den Schadensbearbeitern sitzen, eine Schicht im Lager beobachten, einen Monatsabschluss verfolgen —, statt diese Arbeit als Beschreibung zu erhalten. Die Lücke, die es adressiert, ist spezifisch und zuverlässig: Menschen können Arbeit, die sie fließend beherrschen, nicht akkurat beschreiben. Ein Praktiker, der gebeten wird, seinen Prozess zu erklären, gibt die offizielle Version, lässt die Ausnahmen aus, die ein Drittel seines Tages ausmachen, und erwähnt nie die Workarounds, die er aufgehört hat zu bemerken. Aus solchen Beschreibungen geschriebene Anforderungen sind jedes Mal auf dieselbe charakteristische Weise korrekt und unvollständig. In Legacy-Kontexten hat Eintauchen einen zweiten Nutzen, weil die in einem jahrzehntealten System kodierten Geschäftsregeln oft nirgendwo sonst existieren, und die Menschen, die sie immer noch von Hand anwenden, wo das System versagt, das Nächste sind, was es an verfügbarer Dokumentation gibt.

## How to Apply ◆

> Die Tabellenkalkulation auf dem zweiten Monitor eines Nutzers ist eine Spezifikation dessen, was das System nicht tut, und sie ist aus jeder Distanz unsichtbar.

- **Beobachten Sie, statt zu interviewen.** Sitzen Sie mit jemandem, der seine tatsächliche Arbeit erledigt, für einen erheblichen Zeitblock — mindestens einen halben Tag. Der Punkt ist zu sehen, was sie tun, nicht zu hören, was sie sagen, dass sie tun, und der Unterschied zwischen beiden ist der gesamte Wert.
- **Achten Sie auf die Workarounds**: die Tabellenkalkulation, der Klebezettel, das zweite Fenster daneben geöffnet, der Schritt, bei dem sie konsequent innehalten. Jeder davon ist eine Anforderung, die das aktuelle System nicht erfüllt, und keine davon wird in irgendeiner Anfrage erscheinen.
- **Fragen Sie nach den Ausnahmen**, weil der Routinepfad der ist, der beschrieben wird. „Was passiert, wenn es nicht unkompliziert ist?" öffnet zuverlässig den Teil der Domäne, den die Spezifikation nie abdeckte, und in den meisten Domänen ist die Komplexität in den Ausnahmen zu finden.
- **Schicken Sie die Menschen, die es bauen werden**, nicht nur Analysten. Über einen Mittler übertragenes Verständnis verliert genau die Details, deren Bedeutung im Voraus nicht offensichtlich ist, und das sind die Details, die zählen.
- **Gehen Sie zu einem bedeutsamen Zeitpunkt.** Ein ruhiger Dienstag zeigt den Routinepfad; Monatsende, eine Spitzenzeit oder ein Vorfall zeigt das System unter den Bedingungen, unter denen seine Unzulänglichkeiten tatsächlich etwas kosten.
- **Schreiben Sie das Beobachtete umgehend auf**, einschließlich der Dinge, die Sie nicht verstanden haben. Die Verwirrungen sind so wertvoll wie die Beobachtungen, und beide verblassen innerhalb eines Tages.
- **Lernen Sie das Vokabular und nutzen Sie es exakt.** Domänenbegriffe tragen präzise Unterscheidungen, und ein Entwickler, der zwei davon vermengt, wird ein Modell bauen, das den Unterschied nicht darstellen kann — was der Weg ist, wie Domänenmodelle beginnen, gegen ihre Domäne zu kämpfen.
- **Wiederholen Sie es periodisch**, statt einmal am Projektanfang. Die Arbeit ändert sich, und das Verständnis eines Teams von der Domäne driftet zu dem, was das System gerade tut.
- **Speisen Sie Beobachtungen explizit zurück** zu den Menschen, die Sie beobachtet haben. Es baut die Beziehung auf, korrigiert Ihre Missverständnisse früh und veranlasst sie häufig, etwas zu erwähnen, das sie nicht für erwähnenswert hielten.
- **Beziehen Sie Support- und Betriebspersonal ein.** Sie sehen die Fehler des Systems über viele Nutzer hinweg, was eine Sicht ist, die weder die Entwickler noch ein einzelner Nutzer haben.

## Tradeoffs ⇄

> Eintauchen erzeugt Verständnis, das keine Menge geschriebener Anforderungen vermittelt, auf Kosten von Entwickler- und Praktikerzeit und einer Abhängigkeit von Menschen, die beschäftigt sind.

**Vorteile:**

- Anforderungen werden verstanden statt transkribiert, was die korrekten-aber-nutzlosen Features verhindert, die Spezifikationen zuverlässig produzieren.
- Workarounds werden sichtbar, und jeder davon ist sowohl eine Anforderung als auch eine quantifizierbare laufende Kosten, die die Organisation bereits zahlt.
- Das Domänenmodell verbessert sich, weil die es bauenden Entwickler die Unterscheidungen gesehen haben, die das Vokabular kodiert, statt sie aus Namen abzuleiten.
- Undokumentierte Geschäftsregeln tauchen auf, und in Legacy-Kontexten sind die Menschen, die sie von Hand anwenden, häufig die letzte verbliebene Quelle.
- Die Beziehung zwischen Entwicklern und Nutzern verbessert sich erheblich, was die Qualität jedes nachfolgenden Gesprächs ändert.

**Kosten und Risiken:**

- Es verbraucht die Zeit sowohl der Entwickler als auch der beobachteten Praktiker, und Letztere sind meist beschäftigte Menschen, deren Verfügbarkeit ausgehandelt werden muss.
- Beobachtung ist störend und kann sich wie Überwachung anfühlen, besonders wenn sie vom Management arrangiert wird statt mit der Person vereinbart.
- Was beobachtet wird, ist die Arbeitsweise einer Person, die eigenwillig sein kann. Einen einzelnen Praktiker zu beobachten und zu verallgemeinern erzeugt ein selbstbewusst falsches Modell.
- Physische Präsenz ist für verteilte Teams schwierig und für manche Domänen unmöglich, und Fernbeobachtung verliert viel vom Wert.
- Entwickler können sich übermäßig mit den beobachteten Nutzern identifizieren und für deren spezifische Bedürfnisse gegen die breitere Nutzerpopulation eintreten.

## How It Could Be

Ein Team, das einen Ersatz für ein Frachtbuchungssystem baute, arbeitete anhand einer 40-seitigen, von einem Business-Analysten geschriebenen Spezifikation. Zwei Entwickler verbrachten vor Projektbeginn einen Tag im Buchungsbüro. Die Spezifikation beschrieb einen linearen Prozess: Sendungsdetails eingeben, Spediteur auswählen, bestätigen. Was sie beobachteten, war, dass Buchende drei Browser-Tabs geöffnet hatten, eine gemeinsame Tabellenkalkulation mit Spediteur-Eigenheiten führten, die das System nicht modellierte — welche Spediteure bestimmte Postleitzahlen ablehnten, welche 48 Stunden Vorlauf brauchten, welche telefonisch anders anboten — und etwa jeden vierten Anruf machten, um Preise auszuhandeln, für die das System kein Feld hatte. Nichts davon stand in der Spezifikation, weil der Analyst gefragt hatte, wie der Prozess funktioniert, und ihm gesagt wurde, wie er funktionieren sollte. Der Ersatz wurde um den tatsächlichen Prozess herum neu gestaltet, und die gemeinsame Tabellenkalkulation wurde zu einer modellierten Entität statt eines Workarounds.

Ein zweites Team nutzte Eintauchen, um Geschäftslogik zu bergen, die es nicht aus dem Code extrahieren konnte. Ein Preismodul enthielt eine 600-zeilige Bedingung, die niemand interpretieren konnte, und der ursprüngliche Entwickler hatte 2014 das Unternehmen verlassen. Statt weiter zu lesen, verbrachten zwei Entwickler zwei Tage mit dem Vertriebsteam, beobachteten, wie sie Preise anboten, und fragten, warum jeder Preis war, was er war. Die auftauchenden Regeln — eine Volumenschwelle, die je nach Kundensegment variierte, ein Legacy-Rabatt für Verträge vor 2011, eine regionale Anpassung, die nur auf zwei Länder angewandt wurde — bildeten etwa 400 der 600 Zeilen ab. Die verbleibenden 200 stellten sich heraus, ein Werbeprogramm zu implementieren, das 2016 geendet hatte und das nichts seither erreicht hatte, eine Schlussfolgerung, die das Team mit Logging bestätigte, bevor es sie löschte.
